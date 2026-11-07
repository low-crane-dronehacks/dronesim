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
    ref = torch.tensor([[7.475132465362549, -4.342502052168129e-06, -1.142246127128601],
                        [-1.1265339851379395, -0.18999958038330078, -1.841422438621521]])
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
