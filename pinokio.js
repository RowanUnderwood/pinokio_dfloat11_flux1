module.exports = {
  title: "DFloat11 FLUX.1 Examples",
  description: "Install and run DFloat11 compressed FLUX.1 text-to-image examples (dev and fill-dev models). Requires an NVIDIA GPU with CUDA support.",
  icon: "icon.png",
  menu: async (kernel) => {
    // Check if the environment and key files seem to be installed
    // A simple check for the virtual environment folder
    let installed = await kernel.exists(__dirname, "env");

    if (installed) {
      return [
        {
          html: "<i class='fa-solid fa-play'></i> Run FLUX.1-dev Example",
          href: "run_dev.json",
          params: { target: "_blank" } // Opens terminal in new tab/window
        },
        {
          html: "<i class='fa-solid fa-fill-drip'></i> Run FLUX.1-Fill-dev Example",
          href: "run_fill.json",
          params: { target: "_blank" } // Opens terminal in new tab/window
        },
        {
          html: "<i class='fa-solid fa-folder-open'></i> View Output Images",
          click: async () => {
            // Ensure output directory exists before trying to open
            await kernel.fs.mkdir(kernel.path(__dirname, "output"));
            kernel.shell.open(kernel.path(__dirname, "output"));
          }
        },
        {
          html: "<i class='fa-solid fa-arrows-rotate'></i> Re-install / Update Dependencies",
          href: "install.json"
        },
        {
          html: "<i class='fa-solid fa-trash'></i> Uninstall",
          href: "uninstall.json"
        }
      ];
    } else {
      return [
        {
          html: "<i class='fa-solid fa-download'></i> Install DFloat11 FLUX.1 Environment",
          href: "install.json"
        }
      ];
    }
  }
};