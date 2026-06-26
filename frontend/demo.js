/**
 * Demo Module - Integración del Music Metrics Demo
 * Maneja la apertura, cierre e interacción del modal de demostración.
 */

const demoModal = document.querySelector("#demoModal");
const demoModalOverlay = document.querySelector("#demoModalOverlay");
const closeDemoButton = document.querySelector("#closeDemoButton");
const demoTabButtons = document.querySelectorAll(".demo-tab-button");
const demoTabContents = document.querySelectorAll(".demo-tab-content");

function animateModalEntry() {
  const modalContent = document.querySelector(".demo-modal-content");
  if (modalContent) {
    modalContent.style.animation = "slideUp 0.3s ease";
  }
}

function activateTab(tabName = "overview") {
  demoTabButtons.forEach((button) => button.classList.remove("active"));
  demoTabContents.forEach((content) => content.classList.remove("active"));

  const activeButton = document.querySelector(`[data-tab="${tabName}"]`);
  const activeContent = document.querySelector(`#tab-${tabName}`);

  if (activeButton) activeButton.classList.add("active");
  if (activeContent) activeContent.classList.add("active");
}

function openDemo(tabName = "overview") {
  if (!demoModal) return;
  demoModal.removeAttribute("hidden");
  document.body.style.overflow = "hidden";
  activateTab(tabName);
  animateModalEntry();
}

function closeDemo() {
  if (!demoModal) return;
  demoModal.setAttribute("hidden", "");
  document.body.style.overflow = "";
}

// Accesos directos desde el banner, desde el bloque de gráficas y desde enlaces de documentación.
document.querySelectorAll("#openDemoButton, #openDemoFromCharts, [data-demo-tab-link]").forEach((trigger) => {
  trigger.addEventListener("click", (event) => {
    const tabName = trigger.getAttribute("data-demo-tab-link") || trigger.getAttribute("data-tab") || "overview";
    event.preventDefault();
    openDemo(tabName);
  });
});

if (closeDemoButton) {
  closeDemoButton.addEventListener("click", closeDemo);
}

if (demoModalOverlay) {
  demoModalOverlay.addEventListener("click", closeDemo);
}

demoTabButtons.forEach((button) => {
  button.addEventListener("click", () => {
    activateTab(button.getAttribute("data-tab") || "overview");
  });
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && demoModal && !demoModal.hasAttribute("hidden")) {
    closeDemo();
  }
});

if (demoModal) {
  demoModal.addEventListener("wheel", (event) => {
    if (event.target.closest(".demo-modal-body")) return;
    event.preventDefault();
  });
}

window.demoModule = {
  openDemo,
  closeDemo,
  activateTab,
};

console.log("Demo module cargado correctamente");
