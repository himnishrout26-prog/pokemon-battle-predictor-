const select1 = document.getElementById("select1");
const select2 = document.getElementById("select2");
const sprite1 = document.getElementById("sprite1");
const sprite2 = document.getElementById("sprite2");
const predictBtn = document.getElementById("predictBtn");
const errorMsg = document.getElementById("errorMsg");
const resultCard = document.getElementById("resultCard");

let pokemonList = [];
let pokemonByName = {};

function setSprite(imgEl, name) {
  const p = pokemonByName[name];
  imgEl.src = (p && p.sprite_url) || "";
}

async function loadPokemon() {
  const res = await fetch("/api/pokemon");
  pokemonList = await res.json();
  pokemonByName = Object.fromEntries(pokemonList.map((p) => [p.name, p]));

  for (const p of pokemonList) {
    const opt1 = document.createElement("option");
    opt1.value = p.name;
    opt1.textContent = p.name;
    select1.appendChild(opt1);

    const opt2 = document.createElement("option");
    opt2.value = p.name;
    opt2.textContent = p.name;
    select2.appendChild(opt2);
  }

  // sensible defaults so the page isn't empty on load
  const default1 = pokemonByName["Charizard"] || pokemonList[0];
  const default2 = pokemonByName["Venusaur"] || pokemonList[1];
  select1.value = default1.name;
  select2.value = default2.name;
  setSprite(sprite1, default1.name);
  setSprite(sprite2, default2.name);
}

select1.addEventListener("change", () => setSprite(sprite1, select1.value));
select2.addEventListener("change", () => setSprite(sprite2, select2.value));

predictBtn.addEventListener("click", async () => {
  errorMsg.textContent = "";
  resultCard.classList.add("hidden");

  const p1 = select1.value;
  const p2 = select2.value;

  if (p1 === p2) {
    errorMsg.textContent = "Pick two different Pokemon.";
    return;
  }

  predictBtn.disabled = true;
  predictBtn.textContent = "Predicting...";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pokemon1: p1, pokemon2: p2 }),
    });
    const data = await res.json();

    if (!res.ok) {
      errorMsg.textContent = data.error || "Something went wrong.";
      return;
    }

    renderResult(data);
  } catch (e) {
    errorMsg.textContent = "Could not reach the server. Is the Flask app running?";
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "\u2694\ufe0f Predict winner";
  }
});

function renderResult(data) {
  document.getElementById("probName1").textContent = data.pokemon1;
  document.getElementById("probName2").textContent = data.pokemon2;
  document.getElementById("probPct1").textContent = `${Math.round(data.win_prob_1 * 100)}%`;
  document.getElementById("probPct2").textContent = `${Math.round(data.win_prob_2 * 100)}%`;
  document.getElementById("probFill1").style.width = `${data.win_prob_1 * 100}%`;
  document.getElementById("probFill2").style.width = `${data.win_prob_2 * 100}%`;
  document.getElementById("winnerName").textContent = data.winner;

  renderMoveBlock("moveBlock1", data.pokemon1, data.move_analysis.pokemon1);
  renderMoveBlock("moveBlock2", data.pokemon2, data.move_analysis.pokemon2);

  const list = document.getElementById("factorsList");
  list.innerHTML = "";
  for (const factor of data.top_factors) {
    const li = document.createElement("li");
    const favors = factor.contribution > 0 ? data.pokemon1 : data.pokemon2;
    li.innerHTML = `<span>${factor.label}</span><span class="favor">favors ${favors}</span>`;
    list.appendChild(li);
  }

  resultCard.classList.remove("hidden");
}

const EFFECTIVENESS_CLASS = {
  "Super effective": "eff-super",
  "Not very effective": "eff-weak",
  "No effect": "eff-none",
  "Normal effectiveness": "eff-normal",
};

function renderMoveBlock(elId, pokemonName, move) {
  const el = document.getElementById(elId);
  const effClass = EFFECTIVENESS_CLASS[move.effectiveness] || "eff-normal";
  el.innerHTML = `
    <span class="move-name">${pokemonName}</span>
    <span class="move-type">${capitalize(move.move_type)}-type (${move.category})</span>
    <span class="move-eff ${effClass}">${move.effectiveness}</span>
    <span class="move-dmg">~${move.estimated_damage} dmg</span>
  `;
}

function capitalize(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

loadPokemon();
