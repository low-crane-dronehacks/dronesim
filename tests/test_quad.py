import torch

from dronehacks.quad import TorchQuad


def _fly(steps=20):
    torch.manual_seed(0)
    q = TorchQuad(2, 'cpu', v_max=(5, 3, 2))
    q.reset_idx(torch.arange(2), torch.tensor([[0., 0., -1.], [1., -1., -0.5]]), torch.tensor([0., 1.0]))
    a = torch.tensor([[1.0, 0., 0., 0.], [0., 0.5, -0.5, 0.3]])
    for _ in range(steps):
        q.step_control(a)
    return q


def test_trajectory_matches_reference_plant():
    """Pinned reference trajectory for TorchQuad."""
    q = _fly()
    ref = torch.tensor([[7.2029948234558105, -2.6603984224493615e-06, -1.0788695812225342],
                        [-1.1929333209991455, -0.2770686149597168, -1.8322824239730835]])
    assert torch.allclose(q.pos_ned, ref, atol=1e-5)


def test_hover_holds_position():
    torch.manual_seed(0)
    q = TorchQuad(4, 'cpu')
    start = torch.tensor([[0., 0., -1.]]).repeat(4, 1)
    q.reset_idx(torch.arange(4), start, torch.zeros(4))
    for _ in range(30):
        q.step_control(torch.zeros(4, 4))
    assert (q.pos_ned - start).abs().max() < 0.05


def test_body_frame_actions_follow_heading():
    """Forward is along the drone's heading: facing east (yaw +90 deg), +forward flies east."""
    torch.manual_seed(0)
    q = TorchQuad(1, 'cpu', v_max=(2, 1, 1))
    q.reset_idx(torch.arange(1), torch.tensor([[0., 0., -1.]]), torch.tensor([torch.pi / 2]))
    for _ in range(20):
        q.step_control(torch.tensor([[1., 0., 0., 0.]]))
    p = q.pos_ned[0]
    assert p[1] > 2.5 and abs(p[0]) < 0.2


def test_ground_reaction_blocks_sinking():
    torch.manual_seed(0)
    q = TorchQuad(1, 'cpu')
    q.reset_idx(torch.arange(1), torch.tensor([[0., 0., -0.08]]), torch.zeros(1))
    q.grounded = torch.tensor([True])
    for _ in range(10):
        q.step_control(torch.tensor([[0., 0., 1., 0.]]))      # full descent command
    assert -q.pos_ned[0, 2] >= 0.08 - 1e-6


def test_grounded_drone_cannot_slide():
    """Friction: a drone resting on the ground does not skid sideways; it has to lift off to move."""
    torch.manual_seed(0)
    q = TorchQuad(1, 'cpu', v_max=(5, 3, 2))
    q.reset_idx(torch.arange(1), torch.tensor([[0., 0., -0.08]]), torch.zeros(1))
    q.grounded = torch.tensor([True])
    for _ in range(10):
        q.step_control(torch.tensor([[0.2, 0.2, 1., 0.]]))     # pushed sideways while held down
        assert -q.pos_ned[0, 2] <= 0.08 + 1e-6
    assert q.pos_ned[0, :2].abs().max() < 1e-4


def test_battery_sag_reduces_punchout_and_drains_soc():
    """PX4-style 4S battery model: punch-out sags terminal voltage, drains SoC, and caps low-SoC climb."""
    torch.manual_seed(0)
    q = TorchQuad(2, 'cpu', v_max=(5, 3, 4))
    q.reset_idx(torch.arange(2), torch.tensor([[0., 0., -1.], [0., 0., -1.]]), torch.zeros(2))
    # Give both envs identical controller gains and motor tau; env 0 = fresh low-IR pack, env 1 = low high-IR pack
    q.controller.K_vel[1] = q.controller.K_vel[0]
    q.controller.K_rot[1] = q.controller.K_rot[0]
    q.controller.K_angvel[1] = q.controller.K_angvel[0]
    q.motor_tau[1] = q.motor_tau[0]
    q.batt_soc[0], q.batt_r_int[0] = 1.0, 0.018
    q.batt_soc[1], q.batt_r_int[1] = 0.2, 0.035
    v_oc = 14.0 + 2.2 * q.batt_soc + 0.6 * q.batt_soc.pow(3)
    q.batt_v_term = (v_oc - q._hover_i * q.batt_r_int).clamp_min(11.0)
    q.batt_v_est = q.batt_v_term.clone()
    v_init = q.batt_v_term.clone()
    soc_init = q.batt_soc.clone()

    for _ in range(10):
        q.step_control(torch.tensor([[0., 0., -1., 0.], [0., 0., -1., 0.]]))  # full climb punch-out

    assert (q.batt_soc < soc_init).all()
    assert (q.batt_v_term < v_init).all()
    # Fresh pack climbs higher (more negative z_ned) than sagged low pack
    assert -q.pos_ned[0, 2] > -q.pos_ned[1, 2]

