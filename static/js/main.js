// main.js - page behaviour only. All NLP analysis happens in the Python backend.

function toggleNav() {
  document.getElementById("navlinks").classList.toggle("open");
}

// Fill the home page form with an example review.
function fillExample() {
  const movie = document.getElementById("movie");
  const review = document.getElementById("review");
  if (movie) movie.value = "Sample Movie";
  if (review) {
    review.value =
      "The acting was brilliant and the soundtrack is beautiful, but the second half felt slow " +
      "and the ending was a disappointment. Christopher Nolan style visuals kept me excited though.";
    review.dispatchEvent(new Event("input"));
  }
}

// Character counter on the home page.
const reviewBox = document.getElementById("review");
const counter = document.getElementById("charcount");
if (reviewBox && counter) {
  const update = () => (counter.textContent = reviewBox.value.length);
  reviewBox.addEventListener("input", update);
  update();
}

// Show the loading spinner when a form is submitted.
document.querySelectorAll("form.js-loading").forEach((form) => {
  form.addEventListener("submit", () => {
    const box = document.getElementById("loading");
    if (box) box.hidden = false;
  });
});
// Hide the spinner if the user comes back with the browser's back button.
window.addEventListener("pageshow", () => {
  const box = document.getElementById("loading");
  if (box) box.hidden = true;
});

// NLP Lab tabs.
const tabBar = document.getElementById("labtabs");
if (tabBar) {
  const buttons = tabBar.querySelectorAll("button");
  const panels = document.querySelectorAll("[data-panel]");
  function show(name) {
    buttons.forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
    panels.forEach((p) => p.classList.toggle("active", p.dataset.panel === name));
  }
  buttons.forEach((b) => b.addEventListener("click", () => show(b.dataset.tab)));
  show(tabBar.dataset.active || "preprocess");
}
