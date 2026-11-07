# Your companion, browser and laptop

The browser editor, Learn notebook and companion terminal use the same
persistent project. Open Workbench and press Ctrl+` for the native terminal.
The project path is `/var/lib/flyx-lab/workspace`. Code and package work happens
inside the companion container; the simulator host remains managed separately.

In the website, open Session → **Use your own terminal** for the downloadable
OpenSSH client and its pinned dependencies. Use the exact workspace origin,
not the administrator/root site. Keep the SSH private key on your laptop and
hold your team's editing control while using a remote coding agent. The
transport is HTTPS/WebSocket on 443; no public worker SSH port is required.

The legacy downloaded helper uses password authentication. Portal-only accounts
must use the browser's native terminal until the site provides a separately
qualified portal-authenticated laptop connection flow; your email code is not
a lab password. Never extract browser cookies or paste them into an agent.
This limitation is explicit rather than offering a nonworking password command.

Give an agent `AI_CONTEXT.md` and `CLAUDE.md` (or `AGENTS.md`) from this project.
No special simulator start command belongs in its flight algorithm. Save before
releasing the GPU. Team storage is bounded and expires at the event's configured
cutoff; download important source, notebooks and evidence for long-term retention.
