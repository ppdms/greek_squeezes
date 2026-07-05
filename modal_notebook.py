"""Run greek_squeezes.ipynb on Modal with zero UI configuration.

The hosted Modal Notebook (modal.com/notebooks) makes you click through an
image selector, a volume picker, a secret picker, a GPU selector, and a notebook
upload every time. This file replaces all of that with two coded entrypoints --
image, volume, secret, GPU, and the notebook itself are all declared here:

    modal run --detach modal_notebook.py::jupyter   # interactive JupyterLab,
                                                     #   reachable ONLY through an
                                                     #   SSH tunnel you open from
                                                     #   your own machine -- never a
                                                     #   public URL. Open the
                                                     #   printed localhost URL, then
                                                     #   Run All (or step cells).
    modal run modal_notebook.py::run_all             # fully headless: execute every
                                                     #   code cell in order with
                                                     #   nbclient, stream outputs to
                                                     #   the terminal, persist
                                                     #   artifacts to the volume, then
                                                     #   exit -- no browser, no port.

What each piece removes:
  * image   -> baked here (squeeze_runtime/ + greek_squeezes.ipynb via copy=True,
               plus all runtime deps incl. textdistance + openssh-server). No
               sidebar image picker.
  * volume  -> `greek-squeezes-data` (the volume modal_app.py already populates with
               the ~21 GB of prepared artifacts) is mounted at
               /mnt/greek-squeezes-data, which the notebook's bootstrap cell
               auto-detects, so it reads prepared artifacts and skips the
               download. No volume picker.
  * secret  -> `huggingface-token` injects HF_TOKEN (only used if the volume is
               empty and the bucket sync runs); `ssh-authorized-key` injects your
               SSH public key for the tunnel. No secret picker.
  * GPU     -> T4 (light: enough for inference if artifacts are missing, idle when all
               artifacts are cached — 3.5x cheaper than the A100 the pipeline trains on).
  * upload  -> the .ipynb is baked into the image at /root/greek_squeezes.ipynb.
               No upload.

Security model for `jupyter`: Jupyter binds to the container's 127.0.0.1 only
-- it is NOT exposed on any URL. An sshd runs in the container, authorized with
YOUR public key (from the `ssh-authorized-key` secret); its port is exposed via
an unencrypted Modal TCP tunnel. That TCP endpoint is internet-reachable, but
sshd rejects every connection not signed by your matching private key, so only
you can get in. You then `ssh -L 8888:localhost:8888` to forward your local
:8888 to the container's localhost :8888 and open http://localhost:8888. There
is no shared-secret URL, no token in any URL, and no public Jupyter surface.

REPO_ROOT points at /root via the image env, where squeeze_runtime/ is baked, so
the notebook finds its runtime package without cloning. Requires Modal client
>= 0.66.40 (for add_local_file). Run from this repo dir so add_local_dir/file
can find squeeze_runtime/ and greek_squeezes.ipynb.

One-time setup -- create the SSH secret from your public key (any key you can
ssh with locally; ed25519 recommended):

    modal secret create ssh-authorized-key \\
      SSH_AUTHORIZED_KEY="$(cat ~/.ssh/id_ed25519.pub)"

Notebook edits: this bakes greek_squeezes.ipynb from THIS repo at deploy time.
After editing the notebook elsewhere (e.g. in Colab), re-sync the repo copy
before re-running, so the bake picks up the latest version.
"""
import modal

VOL_MOUNT = "/mnt/greek-squeezes-data"
NB_REMOTE = "/root/greek_squeezes.ipynb"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libglib2.0-0", "openssh-server")
    .pip_install(
        "torch>=2.2",
        "transformers>=4.41",
        "accelerate>=0.30",
        "timm>=0.9",
        "einops>=0.7",
        "opencv-python-headless>=4.8",
        "pillow>=10.0",
        "numpy<2.2",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "matplotlib>=3.7",
        "textdistance>=4.5",
        "sentencepiece>=0.1",
        "requests>=2.31",
        "huggingface_hub[hf_xet]>=1.5.0",
        # notebook execution / interactive server
        "jupyterlab>=4.0",
        "nbclient>=0.10",
        "ipykernel>=6.0",
    )
    .add_local_dir("squeeze_runtime", "/root/squeeze_runtime", copy=True)
    .add_local_file("greek_squeezes.ipynb", NB_REMOTE, copy=True)
    .env({"MODAL_REPO_ROOT": "/root"})
)

app = modal.App("greek-squeezes-notebook")

vol = modal.Volume.from_name("greek-squeezes-data", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-token")
ssh_secret = modal.Secret.from_name("ssh-authorized-key")

# Shared per-function config: image + volume + GPU + timeout (secrets added per
# function, since only `jupyter` needs the SSH key. Change gpu to "a100" / "h100" /
# "l40s" if you need a heavier card for retraining.
_COMMON = dict(
    image=image,
    volumes={VOL_MOUNT: vol},
    gpu="T4",
    timeout=60 * 60 * 8,  # 8h ceiling for long pipeline runs / interactive sessions
)


@app.function(**_COMMON, secrets=[hf_secret, ssh_secret])
def jupyter():
    """Private interactive Jupyter, reachable only through an SSH tunnel.

    Jupyter is bound to the container's 127.0.0.1:8888 and is never exposed on a
    public URL. An sshd is started inside the container, authorized with the
    public key from the `ssh-authorized-key` secret, and its port is exposed via
    an unencrypted Modal TCP tunnel. The tunnel endpoint is internet-reachable,
    but sshd rejects any connection not signed by your matching private key, so
    only you can get in. You then forward your local :8888 to the container's
    localhost :8888 over SSH and open http://localhost:8888 in your browser.

    One-time secret setup:
        modal secret create ssh-authorized-key \\
          SSH_AUTHORIZED_KEY="$(cat ~/.ssh/id_ed25519.pub)"

    Run (detached so a stray Ctrl-C in this terminal doesn't kill it):
        modal run --detach modal_notebook.py::jupyter

    Then copy the printed `ssh ... -L 8888:localhost:8888` command into a local
    terminal, run it, and open the printed http://localhost:8888 URL. Stop the
    container with `modal app stop greek-squeezes-notebook` when done.
    """
    import os, secrets, subprocess, pathlib

    # --- sshd: authorize the caller's public key, start the daemon ---
    ssh_dir = pathlib.Path("/root/.ssh")
    ssh_dir.mkdir(parents=True, exist_ok=True)
    pubkey = os.environ.get("SSH_AUTHORIZED_KEY", "").strip()
    if not pubkey:
        raise RuntimeError(
            "SSH_AUTHORIZED_KEY missing. Create the secret with:\n"
            '  modal secret create ssh-authorized-key '
            'SSH_AUTHORIZED_KEY="$(cat ~/.ssh/id_ed25519.pub)"'
        )
    (ssh_dir / "authorized_keys").write_text(pubkey + "\n")
    (ssh_dir / "authorized_keys").chmod(0o600)
    pathlib.Path("/run/sshd").mkdir(parents=True, exist_ok=True)
    subprocess.run(["ssh-keygen", "-A"], check=True)
    # Start sshd as a daemon: root key login only, no passwords.
    subprocess.run(
        ["/usr/sbin/sshd",
         "-o", "PermitRootLogin=prohibit-password",
         "-o", "PasswordAuthentication=no",
         "-o", "PubkeyAuthentication=yes",
         "-o", "AllowUsers=root"],
        check=True)

    # --- Jupyter: localhost only, random token, in the background ---
    token = secrets.token_urlsafe(20)
    nb_proc = subprocess.Popen(
        ["jupyter", "lab", "--no-browser", "--allow-root",
         "--ip=127.0.0.1", "--port=8888",
         f"--ServerApp.token={token}",
         "--ServerApp.allow_remote_access=0"],
        env={**os.environ, "JUPYTER_TOKEN": token, "SHELL": "/bin/bash"})

    # --- expose sshd port (public TCP, key-gated) and print the ssh command ---
    with modal.forward(22, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        ssh_cmd = (
            "ssh -N -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
            f"-p {port} root@{host} -L 8888:localhost:8888"
        )
        print("\n>>> 1) run this in a local terminal to open the tunnel:")
        print(f"    {ssh_cmd}")
        print(">>> 2) then open in your browser (only reachable via the tunnel):")
        print(f"    http://localhost:8888/lab?token={token}")
        print(f">>> notebook on disk: {NB_REMOTE}")
        print(">>> stop the container with: modal app stop greek-squeezes-notebook\n")
        # Block until the app is stopped / interrupted; keeps the tunnel + jupyter up.
        nb_proc.wait()


@app.function(**_COMMON, secrets=[hf_secret])
def run_all():
    """Headlessly execute every code cell of greek_squeezes.ipynb in order.

    `modal run modal_notebook.py::run_all` runs the whole pipeline with no
    browser and no exposed port: cells execute in one kernel, artifacts are
    written to the mounted volume (persisted on exit), and cell outputs /
    tracebacks stream to the terminal. Use this once the notebook is stable and
    you just want it run end-to-end on the GPU.
    """
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.read(NB_REMOTE, as_version=4)
    client = NotebookClient(
        nb, timeout=7200, kernel_name="python3",
        resources={"metadata": {"path": "/root"}},
    )
    client.execute()
    out = f"{VOL_MOUNT}/greek_squeezes__executed.ipynb"
    nbformat.write(nb, out)
    print(f"done; executed notebook with outputs -> {out}")