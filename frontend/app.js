// Resolución automática del backend (sin configuración manual):
//   1) ?backend=<url> en la URL → se guarda (enlace "1-click" desde Colab).
//   2) localStorage.hybrid_api_base si existe.
//   3) En localhost (desarrollo) → backend local 127.0.0.1:8100.
//   4) Desplegado (Netlify/Vercel) → mismo origen "" ; Netlify hace de proxy
//      de /api/* hacia el backend real (ver netlify.toml). Evita CORS y
//      contenido mixto porque la llamada sale del propio dominio https.
function resolveApiBase() {
  const params = new URLSearchParams(location.search);
  // ?cloud=<url> preconfigura el backend GPU de Colab (Paso 4) en un clic.
  const cloudFromQuery = params.get("cloud");
  if (cloudFromQuery) localStorage.setItem("hybrid_cloud_base", cloudFromQuery.replace(/\/+$/, ""));
  const fromQuery = params.get("backend");
  if (fromQuery) localStorage.setItem("hybrid_api_base", fromQuery.replace(/\/+$/, ""));
  const stored = localStorage.getItem("hybrid_api_base");
  if (stored) return stored;
  if (["localhost", "127.0.0.1"].includes(location.hostname)) return "http://127.0.0.1:8100";
  return ""; // mismo origen → proxy de Netlify
}
const API_BASE = resolveApiBase();

// Backend GPU opcional en la nube (Colab + túnel). Solo afecta a la generación
// con modelos pre-entrenados del Paso 4; el resto del pipeline sigue en local.
function cloudBase() {
  return (localStorage.getItem("hybrid_cloud_base") || "").replace(/\/+$/, "");
}

const state = {
  projects: [],
  selectedProjectId: "",
  selectedProject: null,
  resources: {
    token_manifests: [],
    token_models: [],
    generations: [],
    rankings: [],
    renders: [],
    blends: [],
    jamendo_catalogs: [],
    token_vae: { models: [], embeddings: [] },
  },
  presets: {
    training: {},
    generation: {},
  },
  activeJobId: "",
  polling: null,
  busy: false,
  lastFocusedStepId: "",
  stepsInitialized: false,
  lastJobRunning: false,
  benchmark: { loaded: false, loading: false },
  evaluationAvailability: null,
  evaluationSources: null,
  evaluations: [],
  evaluationReports: {},
  evaluationError: "",
};

const els = {
  apiStatus: document.querySelector("#apiStatus"),
  nextStep: document.querySelector("#nextStep"),
  refreshButton: document.querySelector("#refreshButton"),
  projectSelect: document.querySelector("#projectSelect"),
  projectName: document.querySelector("#projectName"),
  createProjectButton: document.querySelector("#createProjectButton"),
  projectMeta: document.querySelector("#projectMeta"),
  jobStatus: document.querySelector("#jobStatus"),
  jobProgress: document.querySelector("#jobProgress"),
  activityPanel: document.querySelector("#jobProgress")?.closest(".panel") || null,
  jobMessage: document.querySelector("#jobMessage"),
  jobEvents: document.querySelector("#jobEvents"),
  steps: document.querySelector("#steps"),
  toggleAdvanced: document.querySelector("#toggleAdvanced"),
  localMidiPanel: document.querySelector("#localMidiPanel"),
  buildCatalogButton: document.querySelector("#buildCatalogButton"),
  cleanMidiButton: document.querySelector("#cleanMidiButton"),
  tokenizeCatalogButton: document.querySelector("#tokenizeCatalogButton"),
  exportInputTokensButton: document.querySelector("#exportInputTokensButton"),
  exportOutputTokensButton: document.querySelector("#exportOutputTokensButton"),
  exportInputTokenSelect: document.querySelector("#exportInputTokenSelect"),
  exportOutputTokenSelect: document.querySelector("#exportOutputTokenSelect"),
  trainTokenModelButton: document.querySelector("#trainTokenModelButton"),
  generateTokensButton: document.querySelector("#generateTokensButton"),
  generateRankedButton: document.querySelector("#generateRankedButton"),
  renderMidiButton: document.querySelector("#renderMidiButton"),
  renderLayersButton: document.querySelector("#renderLayersButton"),
  analyzeMidiButton: document.querySelector("#analyzeMidiButton"),
  blendEmbeddingsButton: document.querySelector("#blendEmbeddingsButton"),
  trainingPresetSelect: document.querySelector("#trainingPresetSelect"),
  generationPresetSelect: document.querySelector("#generationPresetSelect"),
  tokenManifestSelect: document.querySelector("#tokenManifestSelect"),
  tokenModelSelect: document.querySelector("#tokenModelSelect"),
  tokenVaeEmbeddingSelect: document.querySelector("#tokenVaeEmbeddingSelect"),
  generationSelect: document.querySelector("#generationSelect"),
  jamendoCatalogSelect: document.querySelector("#jamendoCatalogSelect"),
  jamendoGenreSelect: document.querySelector("#jamendoGenreSelect"),
  jamendoCatalogSummary: document.querySelector("#jamendoCatalogSummary"),
  rankingList: document.querySelector("#rankingList"),
  downloadList: document.querySelector("#downloadList"),
  audioPlayers: document.querySelector("#audioPlayers"),
  jamendoJson: document.querySelector("#jamendoJson"),
  datasetJson: document.querySelector("#datasetJson"),
  outputJson: document.querySelector("#outputJson"),
  modelJson: document.querySelector("#modelJson"),
  renderJson: document.querySelector("#renderJson"),
  blendJson: document.querySelector("#blendJson"),
};

const GEN_ENGINES = [
  {
    value: "custom",
    badge: "Propio",
    badgeClass: "custom",
    name: "Transformer Propio",
    hfId: "MTG-Jamendo → Tokens",
    note: "Entrena desde cero con tus clips",
  },
  {
    value: "musicgen-small",
    badge: "HF",
    badgeClass: "hf",
    name: "MusicGen-small",
    hfId: "facebook/musicgen-small",
    note: "~300 MB · Rápido · Texto→Audio",
  },
  {
    value: "musicgen-medium",
    badge: "HF",
    badgeClass: "hf",
    name: "MusicGen-medium",
    hfId: "facebook/musicgen-medium",
    note: "~1.5 GB · Mejor calidad · Texto→Audio",
  },
  {
    value: "audioldm2",
    badge: "HF",
    badgeClass: "hf",
    name: "AudioLDM2",
    hfId: "cvssp/audioldm2",
    note: "~3 GB · Difusión latente",
  },
  {
    value: "stable-audio-open",
    badge: "HF",
    badgeClass: "hf",
    name: "Stable Audio Open",
    hfId: "stabilityai/stable-audio-open-1.0",
    note: "~3 GB · Stability AI · Difusión",
  },
];

function renderEngineCards() {
  return `
    <p class="field-section-label">Motor de generación</p>
    <div class="model-card-grid" id="genEngineGrid">
      ${GEN_ENGINES.map((e) => `
        <label class="model-card${e.value === "custom" ? " is-selected" : ""}" data-engine="${escapeHtml(e.value)}">
          <input type="radio" name="genEngine" value="${escapeHtml(e.value)}"${e.value === "custom" ? " checked" : ""} hidden />
          <span class="model-card-badge model-card-badge--${e.badgeClass}">${escapeHtml(e.badge)}</span>
          <strong class="model-card-name">${escapeHtml(e.name)}</strong>
          <span class="model-card-id">${escapeHtml(e.hfId)}</span>
          <span class="model-card-note">${escapeHtml(e.note)}</span>
        </label>
      `).join("")}
    </div>
  `;
}

// Paso independiente de generación con los 4 modelos pre-entrenados + conexión a
// la GPU de la nube (Colab). Reutiliza los IDs del panel de nube; no colisiona
// porque jamendo-train (que también los usa) solo existe en modo jamendo.
function renderPretrainedStep() {
  const hf = GEN_ENGINES.filter((e) => e.value !== "custom");
  const stored = localStorage.getItem("hybrid_cloud_base") || "";
  return `
    <p class="field-section-label">Modelo de generación</p>
    <div class="model-card-grid" id="genEngineGrid">
      ${hf
        .map(
          (e, i) => `
        <label class="model-card${i === 0 ? " is-selected" : ""}" data-engine="${escapeHtml(e.value)}">
          <input type="radio" name="genEngine" value="${escapeHtml(e.value)}"${i === 0 ? " checked" : ""} hidden />
          <span class="model-card-badge model-card-badge--${e.badgeClass}">${escapeHtml(e.badge)}</span>
          <strong class="model-card-name">${escapeHtml(e.name)}</strong>
          <span class="model-card-id">${escapeHtml(e.hfId)}</span>
          <span class="model-card-note">${escapeHtml(e.note)}</span>
        </label>
      `,
        )
        .join("")}
    </div>
    <div class="data-sources-intro">
      <p class="field-note">Genera audio para los 100 prompts de <code>testset_metadata.csv</code> y los guarda en <code>wavs/&lt;modelo&gt;/</code>. Después úsalos en la pestaña Gráficas.</p>
    </div>
    <div class="cloud-panel">
      <div class="cloud-row">
        <span class="cloud-dot${stored ? " is-on" : ""}" id="cloudDot"></span>
        <label class="field" style="flex:1">
          <span>Backend en la nube (GPU) — Colab</span>
          <input id="cloudBaseInput" value="${escapeHtml(stored)}" placeholder="https://xxxx.trycloudflare.com" />
          <small>Pega la URL del túnel que imprime tu notebook de Colab. Vacío = generar en esta máquina (lento).</small>
        </label>
        <button class="secondary-button" data-cycle-action="connect-cloud" type="button">Conectar</button>
      </div>
      <p class="cloud-status" id="cloudStatus">${
        stored
          ? `Configurado: ${escapeHtml(stored)} (pulsa Conectar para verificar)`
          : "Sin conectar — la generación correrá localmente."
      }</p>
    </div>
    <div id="cloudDownload"></div>
  `;
}

const stepDefinitions = [
  {
    id: "import",
    title: "Importar audio",
    help: "Elige el banco de datos y decide si usas pistas ya descargadas o descargas nuevas.",
    requirement: () => Boolean(state.selectedProject),
    complete: () =>
      Boolean(state.selectedProject?.source?.normalized || localStorage.getItem("hybrid_selected_jamendo_catalog_path")),
    actionLabel: "Importar audio individual",
    secondaryLabel: "Usar pistas seleccionadas",
    renderFields: () => `
      <div class="data-sources-intro">
        <p class="field-note"><strong>Bancos de datos disponibles</strong></p>
        <ul class="data-sources-list">
          <li><b>MTG-Jamendo</b> — música libre con licencia Creative Commons, organizada
            por género. Es la fuente para <em>entrenar</em> el modelo. Puedes
            <b>usar las pistas ya descargadas</b> (selector de abajo) o
            <b>descargar nuevas</b> (panel “Descargar más pistas”).</li>
          <li><b>MusicCaps</b> — 5,521 clips de YouTube de 10 s con captions de músicos
            profesionales. Es la fuente del <em>banco de pruebas</em> para
            <em>evaluar</em> (FAD/CLAP/KLD/KAD). Se construye con
            <code>python scripts/sample_testset.py</code> (100 clips, 10 géneros × 10).</li>
          <li><b>Audio individual</b> — una sola canción local para análisis puntual.</li>
        </ul>
      </div>
      <label class="field">
        <span>Audio individual</span>
        <input id="audioPath" placeholder="/Users/tu_usuario/Music/cancion.wav" />
        <small>Opcional. Úsalo si quieres analizar una canción concreta.</small>
      </label>
      <div class="selector-box">
        <div class="panel-heading compact-heading">
          <span>Opción A · Usar pistas ya descargadas (MTG-Jamendo)</span>
          <strong>Selecciona género y cantidad</strong>
        </div>
        <div class="inline-fields">
          <label class="field">
            <span>Catálogo descargado</span>
            <select id="jamendoCatalogSelect"></select>
            <small>Catálogo local de MTG-Jamendo con conteos por género.</small>
          </label>
          <label class="field">
            <span>Pistas por género</span>
            <input id="jamendoTrackLimit" type="number" value="100" min="1" />
            <small>Cantidad máxima que se usará de cada género seleccionado.</small>
          </label>
        </div>
        <label class="field">
          <span>Géneros descargados</span>
          <select id="jamendoGenreSelect" multiple size="6"></select>
          <small>Usa Cmd/Ctrl para seleccionar varios géneros.</small>
        </label>
        <p class="muted" id="jamendoCatalogSummary">Aún no hay catálogo seleccionado.</p>
        <details class="advanced">
          <summary>Opción B · Descargar nuevas pistas (MTG-Jamendo) o preparar clips</summary>
          <label class="field">
            <span>Géneros y tags JSON</span>
            <textarea id="jamendoGenreTags" class="text-area">{
  "classical": ["classical", "orchestral", "piano", "strings"],
  "electronic": ["electronic", "techno", "house", "ambient", "edm", "trance"],
  "reggaeton": ["reggae", "latin", "reggaeton"]
}</textarea>
            <small>Solo cambia esto si necesitas descargar otros géneros.</small>
          </label>
          <div class="inline-fields">
            <label class="field">
              <span>Tracks por página</span>
              <input id="jamendoTracksPerPage" type="number" value="200" min="1" />
              <small>Tamaño de cada consulta interna.</small>
            </label>
            <label class="field">
              <span>Máximo por género</span>
              <input id="jamendoMaxTracks" type="number" value="500" min="1" />
              <small>Por defecto descarga hasta 500 pistas por género.</small>
            </label>
          </div>
          <div class="action-row">
            <button class="secondary-button" data-jamendo-action="download" type="button">Descargar pistas</button>
            <button class="secondary-button" data-jamendo-action="metadata" type="button">Solo metadata</button>
          </div>
        </details>
        <details class="advanced">
          <summary>Entrada alternativa: MIDI local por géneros</summary>
          <p class="field-note">Úsalo solo si ya tienes una biblioteca MIDI organizada por carpetas de género.</p>
          <div class="inline-fields">
            <label class="field">
              <span>Carpeta fuente MIDI</span>
              <input id="genreSourceDir" placeholder="/ruta/dataset_midi" />
              <small>Debe contener subcarpetas por género con archivos .mid/.midi.</small>
            </label>
            <label class="field">
              <span>Géneros</span>
              <input id="genreNames" value="genre_a,genre_b,genre_c" />
              <small>Uno o más nombres, separados por comas.</small>
            </label>
            <label class="field">
              <span>Clips por género</span>
              <input id="clipsPerGenre" type="number" value="200" min="1" />
              <small>Cantidad máxima de MIDIs válidos por género.</small>
            </label>
            <label class="field">
              <span>Duración máxima</span>
              <input id="maxClipDuration" type="number" value="10" min="1" />
              <small>Los MIDIs más largos se rechazan del catálogo.</small>
            </label>
            <label class="field">
              <span>Carpeta a limpiar</span>
              <input id="cleanMidiSourceDir" placeholder="/ruta/midis_sin_limpiar" />
              <small>Escanea, valida, deduplica y copia MIDIs limpios.</small>
            </label>
            <label class="field">
              <span>Nombre dataset limpio</span>
              <input id="cleanMidiName" value="clean_midis" />
              <small>Nombre del lote limpio que se guardará.</small>
            </label>
          </div>
          <div class="action-row">
            <button class="secondary-button" data-cycle-action="clean-midi" type="button">Limpiar MIDI</button>
            <button class="secondary-button" data-cycle-action="build-catalog" type="button">Crear catálogo JSON</button>
            <button class="secondary-button" data-cycle-action="augment-midi" type="button">Augmentar MIDI</button>
            <button class="secondary-button" data-cycle-action="tokenize-catalog" type="button">Tokenizar y crear ZIP</button>
          </div>
        </details>
        <pre class="json-box" id="jamendoJson">{}</pre>
      </div>
    `,
    run: () => {
      const sourcePath = document.querySelector("#audioPath")?.value?.trim();
      if (!sourcePath) throw new Error("Indica la ruta local del audio.");
      return postJob("/api/jobs/import-audio", {
        project_id: state.selectedProjectId,
        source_path: sourcePath,
      });
    },
    runSecondary: () => selectJamendoCatalog(),
  },
  {
    id: "jamendo-clips",
    title: "Cortar clips de entrenamiento",
    help: "Convierte las pistas seleccionadas en clips WAV cortos y organizados por género.",
    mode: "jamendo",
    requirement: () => Boolean(localStorage.getItem("hybrid_selected_jamendo_catalog_path")),
    complete: () => Boolean(localStorage.getItem("hybrid_jamendo_clips_catalog_path")),
    actionLabel: "Cortar clips WAV",
    renderFields: () => `
      ${renderJamendoStorageGuide()}
      <div class="inline-fields">
        <label class="field">
          <span>Catálogo seleccionado</span>
          <input id="jamendoCatalogPath" value="${escapeHtml(
            localStorage.getItem("hybrid_selected_jamendo_catalog_path") || "",
          )}" />
          <small>Catálogo creado al elegir género y cantidad de pistas.</small>
        </label>
        <label class="field">
          <span>Duración del clip</span>
          <input id="jamendoClipDuration" type="number" value="20" min="1" />
          <small>Duración objetivo de cada WAV.</small>
        </label>
        <label class="field">
          <span>Salto entre clips</span>
          <input id="jamendoHopDuration" placeholder="Vacío = igual a duración" />
          <small>Menor que la duración produce clips solapados.</small>
        </label>
        <label class="field">
          <span>Máximo por pista</span>
          <input id="jamendoMaxClipsPerTrack" value="1" />
          <small>Para empezar, 1 clip por pista mantiene el proceso controlado.</small>
        </label>
      </div>
    `,
    run: () => prepareJamendoClips(),
  },
  {
    id: "jamendo-process",
    title: "Procesar clips para entrenamiento",
    help: "Extrae features y tokens desde los clips. Esto genera el manifest que después usa el modelo.",
    mode: "jamendo",
    requirement: () => Boolean(localStorage.getItem("hybrid_jamendo_clips_catalog_path")),
    complete: () => Boolean(localStorage.getItem("hybrid_jamendo_token_manifest_path")),
    actionLabel: "Procesar clips",
    renderFields: () => `
      ${renderJamendoStorageGuide()}
      <div class="inline-fields">
        <label class="field">
          <span>Catálogo de clips</span>
          <input id="clipsCatalogPath" value="${escapeHtml(
            localStorage.getItem("hybrid_jamendo_clips_catalog_path") || "",
          )}" />
          <small>Archivo generado al cortar clips WAV.</small>
        </label>
        <label class="field">
          <span>Máximo de clips</span>
          <input id="processMaxClips" placeholder="Vacío = todos" />
          <small>Útil para una prueba rápida antes de procesar todo.</small>
        </label>
      </div>
      <div class="checkbox-row">
        <label><input id="processStems" type="checkbox" /> separar stems opcional</label>
        <label><input id="processTrainingMaterial" type="checkbox" checked /> Crear MIDI y features para entrenamiento</label>
      </div>
      <p class="field-note">Genera MIDI melódico, MIDI de batería, features y tokens. Déjalo activado para entrenar modelos.</p>
      <p class="field-note">Stems usa Demucs y mejora la separación por capas; actívalo si quieres más calidad y puedes esperar más tiempo.</p>
    `,
    run: () => processJamendoClips(),
  },
  {
    id: "jamendo-train",
    title: "Entrenar modelo generativo",
    help: "Usa los tokens procesados para entrenar un Transformer pequeño listo para generar MIDI.",
    mode: "jamendo",
    requirement: () => Boolean(localStorage.getItem("hybrid_jamendo_token_manifest_path")),
    complete: () => Boolean(localStorage.getItem("hybrid_jamendo_model_path")),
    actionLabel: "Entrenar modelo",
    renderFields: () => `
      ${renderEngineCards()}
      <div id="trainCustomSection">
        ${renderJamendoStorageGuide()}
        <div class="inline-fields">
          <label class="field">
            <span>Tokens de entrenamiento</span>
            <input id="stepTokenManifestPath" value="${escapeHtml(
              localStorage.getItem("hybrid_jamendo_token_manifest_path") || "",
            )}" />
            <small>Manifest generado en el paso anterior. Este archivo alimenta el entrenamiento.</small>
          </label>
          <label class="field">
            <span>Nombre del modelo</span>
            <input id="stepTokenModelName" value="jamendo_transformer" />
            <small>Nombre para identificar este entrenamiento.</small>
          </label>
        </div>
        <details class="advanced">
          <summary>Parámetros de entrenamiento</summary>
          <div class="inline-fields">
            <label class="field">
              <span>Épocas</span>
              <input id="stepTransformerEpochs" type="number" value="8" min="1" />
              <small>Más épocas tardan más y pueden aprender mejor.</small>
            </label>
            <label class="field">
              <span>Contexto</span>
              <input id="stepTransformerSequenceLength" type="number" value="128" min="16" />
              <small>Cuántos tokens mira el modelo para predecir el siguiente.</small>
            </label>
          </div>
        </details>
        <details class="advanced">
          <summary>Token-VAE para embeddings latentes</summary>
          <div class="inline-fields">
            <label class="field">
              <span>Tokens para Token-VAE</span>
              <select id="tokenVaeManifestSelect"></select>
              <small>Prioriza manifests creados con "Procesar con Demucs para Token-VAE".</small>
            </label>
            <label class="field">
              <span>Dimensión latente</span>
              <input id="stepTokenVaeLatentDim" type="number" value="32" min="2" />
              <small>Tamaño del embedding musical aprendido desde tokens.</small>
            </label>
            <label class="field">
              <span>Épocas Token-VAE</span>
              <input id="stepTokenVaeEpochs" type="number" value="60" min="1" />
              <small>Entrenamiento de representación, no reemplaza al Transformer.</small>
            </label>
          </div>
          <div class="action-row">
            <button class="secondary-button" data-cycle-action="process-token-vae-demucs" type="button">
              Procesar con Demucs para Token-VAE
            </button>
            <button class="secondary-button" data-cycle-action="train-token-vae" type="button">Entrenar Token-VAE</button>
            <button class="secondary-button" data-cycle-action="encode-token-vae" type="button">Crear embedding Token-VAE</button>
            <button class="secondary-button" data-cycle-action="encode-genre-embeddings" type="button">Crear embeddings por género</button>
          </div>
          <small>El modo Demucs tarda más, pero genera capas MIDI más limpias para embeddings y fusión.</small>
        </details>
      </div>
      <div id="trainPretrainedSection" hidden>
        <div class="data-sources-intro">
          <p class="field-note"><strong>Modelo pre-entrenado</strong> — se descarga desde HuggingFace la primera vez que se usa (~300 MB – 3 GB según el modelo).</p>
          <p class="field-note">Genera audio para los 100 clips del banco de pruebas (<code>testset_metadata.csv</code>) y los guarda en <code>wavs/&lt;modelo&gt;/</code>. Después úsalos en la pestaña Métricas.</p>
        </div>
        <div class="cloud-panel">
          <div class="cloud-row">
            <span class="cloud-dot${localStorage.getItem("hybrid_cloud_base") ? " is-on" : ""}" id="cloudDot"></span>
            <label class="field" style="flex:1">
              <span>Backend en la nube (GPU) — opcional</span>
              <input id="cloudBaseInput" value="${escapeHtml(localStorage.getItem("hybrid_cloud_base") || "")}" placeholder="https://xxxx.trycloudflare.com" />
              <small>Pega la URL que imprime el notebook de Colab. Vacío = generar en esta máquina (CPU/MPS, lento).</small>
            </label>
            <button class="secondary-button" data-cycle-action="connect-cloud" type="button">Conectar</button>
          </div>
          <p class="cloud-status" id="cloudStatus">${
            localStorage.getItem("hybrid_cloud_base")
              ? `Configurado: ${escapeHtml(localStorage.getItem("hybrid_cloud_base"))} (pulsa Conectar para verificar)`
              : "Sin conectar — la generación correrá localmente."
          }</p>
        </div>
        <div class="action-row">
          <button class="secondary-button" data-cycle-action="generate-pretrained" type="button">
            Generar audio (MusicCaps testset → wavs/)
          </button>
        </div>
        <div id="cloudDownload"></div>
      </div>
    `,
    run: () => {
      const engine = document.querySelector('input[name="genEngine"]:checked')?.value || "custom";
      if (engine === "custom") return trainJamendoModel();
      return generatePretrainedModel(engine);
    },
  },
  {
    id: "jamendo-generate",
    title: "Generar música",
    help: "Genera varias versiones con el modelo entrenado, las rankea y deja MIDI por capas.",
    mode: "jamendo",
    requirement: () => Boolean(localStorage.getItem("hybrid_jamendo_model_path")),
    complete: () => Boolean(localStorage.getItem("hybrid_jamendo_ranking_path")),
    actionLabel: "Generar y elegir mejor",
    renderFields: () => `
      ${renderJamendoStorageGuide()}
      <div class="inline-fields">
        <label class="field">
          <span>Modelo entrenado</span>
          <input id="stepTokenModelPath" value="${escapeHtml(
            localStorage.getItem("hybrid_jamendo_model_path") || "",
          )}" />
          <small>Modelo creado en el paso anterior.</small>
        </label>
        <label class="field">
          <span>Nombre de salida</span>
          <input id="stepGenerationName" value="jamendo_generation" />
          <small>Nombre base de la generación.</small>
        </label>
        <label class="field">
          <span>Duración</span>
          <input id="stepGenerationDuration" type="number" value="30" min="1" />
          <small>Duración objetivo en segundos.</small>
        </label>
        <label class="field">
          <span>Versiones a generar</span>
          <input id="stepRankedCandidates" type="number" value="6" min="1" />
          <small>Genera varias opciones y ordena las mejores por métricas MIDI.</small>
        </label>
        <label class="field">
          <span>Género guía</span>
          <input id="stepGenerationGenre" placeholder="electronic, classical, reggaeton" />
          <small>Opcional. Usa un género presente en los tokens de entrenamiento.</small>
        </label>
        <label class="field">
          <span>Seed</span>
          <input id="stepGenerationSeed" placeholder="Vacío = aleatorio" />
          <small>Usa el mismo número para repetir una generación parecida.</small>
        </label>
        <label class="field">
          <span>Tokens máximos</span>
          <input id="stepGenerationMaxTokens" type="number" value="1200" min="32" />
          <small>Más tokens permiten piezas más largas o densas, pero tardan más.</small>
        </label>
        <label class="field">
          <span>Embedding Token-VAE</span>
          <select id="tokenVaeEmbeddingSelect"></select>
          <small>Opcional. Condiciona la generación con un vector latente aprendido.</small>
        </label>
      </div>
      <details class="advanced">
        <summary>Parámetros de generación recomendados</summary>
        <div class="inline-fields">
          <label class="field">
            <span>Temperatura</span>
            <input id="stepGenerationTemperature" type="number" value="0.84" min="0.1" max="2" step="0.01" />
            <small>Controla riesgo/creatividad. 0.75-0.9 suele ser más musical.</small>
          </label>
          <label class="field">
            <span>Top-k</span>
            <input id="stepGenerationTopK" type="number" value="56" min="1" />
            <small>Limita cada decisión a las opciones más probables.</small>
          </label>
          <label class="field">
            <span>Top-p</span>
            <input id="stepGenerationTopP" type="number" value="0.92" min="0.1" max="1" step="0.01" />
            <small>Filtra opciones hasta cubrir una probabilidad acumulada.</small>
          </label>
        </div>
      </details>
      <div class="checkbox-row">
        <label><input id="stepExportLayerMidis" type="checkbox" checked /> crear MIDI por capas</label>
        <label><input id="stepRankedRenderBest" type="checkbox" checked /> renderizar mejor versión</label>
      </div>
      <details class="advanced">
        <summary>Fusión explícita de géneros</summary>
        <div class="inline-fields">
          <label class="field">
            <span>Embeddings por género</span>
            <select id="genreEmbeddingRunSelect"></select>
            <small>Resultado de “Crear embeddings por género”. Contiene un vector latente por género.</small>
          </label>
          <label class="field">
            <span>Nombre de la fusión</span>
            <input id="genreFusionName" value="fusion_generos" />
            <small>Nombre para guardar el embedding híbrido.</small>
          </label>
        </div>
        <label class="field">
          <span>Pesos por género</span>
          <textarea id="genreFusionWeights" rows="4" placeholder='{"electronic": 0.5, "reggaeton": 0.3, "classical": 0.2}'></textarea>
          <small>Indica cuánto aporta cada género. Los pesos se normalizan automáticamente.</small>
        </label>
        <p class="field-note">Mezcla los géneros elegidos y genera candidatas nuevas usando esa mezcla.</p>
        <div class="action-row">
          <button class="primary-button" data-cycle-action="generate-genre-fusion" type="button">Generar música con fusión de géneros</button>
        </div>
        <details class="advanced">
          <summary>Opciones avanzadas de fusión</summary>
          <div class="action-row">
            <button class="secondary-button" data-cycle-action="blend-genre-embeddings" type="button">Solo guardar receta de fusión</button>
            <button class="secondary-button" data-cycle-action="compare-fusions" type="button">Comparar fusiones guardadas</button>
          </div>
        </details>
      </details>
    `,
    run: () => generateJamendoRanked(),
  },
  {
    id: "stems",
    title: "Separar stems",
    help: "Usa Demucs para separar batería, bajo, voz y otros. Es el paso más pesado del pipeline.",
    requirement: () => Boolean(state.selectedProject?.source?.normalized),
    complete: () => Boolean(state.selectedProject?.stems?.files),
    actionLabel: "Separar stems",
    renderFields: () => `
      <details class="advanced">
        <summary>Opciones avanzadas</summary>
        <div class="inline-fields">
          <label class="field">
            <span>Modelo Demucs</span>
            <input id="demucsModel" value="htdemucs" />
            <small>Modelo base recomendado para separación general.</small>
          </label>
          <label class="field">
            <span>Dispositivo</span>
            <select id="demucsDevice">
              <option value="auto">auto</option>
              <option value="cpu">cpu</option>
              <option value="mps">mps</option>
              <option value="cuda">cuda</option>
            </select>
            <small>Usa auto salvo que necesites forzar CPU/GPU.</small>
          </label>
        </div>
      </details>
    `,
    run: () =>
      postJob("/api/jobs/separate-stems", {
        project_id: state.selectedProjectId,
        model_name: document.querySelector("#demucsModel")?.value || "htdemucs",
        device: document.querySelector("#demucsDevice")?.value || "auto",
      }),
  },
  {
    id: "midi",
    title: "Transcribir MIDI por capas",
    help: "Convierte stems en MIDI separado: Basic Pitch para capas melódicas y onsets para batería.",
    requirement: () => Boolean(state.selectedProject?.stems?.files),
    complete: () => Boolean(state.selectedProject?.midis?.melodic || state.selectedProject?.midis?.drums),
    actionLabel: "Transcribir melodía",
    secondaryLabel: "Transcribir batería",
    renderFields: () => `
      <details class="advanced">
        <summary>Opciones avanzadas</summary>
        <div class="inline-fields">
          <label class="field">
            <span>Stems melódicos</span>
            <input id="melodicStems" value="bass,vocals,other" />
            <small>Lista separada por comas.</small>
          </label>
          <label class="field">
            <span>BPM batería</span>
            <input id="drumBpm" placeholder="Vacío = detectar" />
            <small>Opcional. Ayuda a cuantificar la batería.</small>
          </label>
        </div>
      </details>
    `,
    run: () =>
      postJob("/api/jobs/transcribe-melodic", {
        project_id: state.selectedProjectId,
        stems: splitCsv(document.querySelector("#melodicStems")?.value || "bass,vocals,other"),
      }),
    runSecondary: () => {
      const bpm = Number.parseFloat(document.querySelector("#drumBpm")?.value || "");
      return postJob("/api/jobs/transcribe-drums", {
        project_id: state.selectedProjectId,
        bpm: Number.isFinite(bpm) ? bpm : null,
      });
    },
  },
  {
    id: "features",
    title: "Extraer features musicales",
    help: "Calcula métricas medibles del audio y los MIDI: tempo, densidad, notas, pitch, ritmo y capas.",
    requirement: () => Boolean(state.selectedProject?.source?.normalized || state.selectedProject?.midis),
    complete: () => Boolean(state.selectedProject?.features?.path),
    actionLabel: "Extraer features",
    renderFields: () => "",
    run: () =>
      postJob("/api/jobs/extract-features", {
        project_id: state.selectedProjectId,
        include_audio: true,
        include_midis: true,
      }),
  },
  {
    id: "vae",
    title: "Entrenar VAE",
    help: "Entrena un modelo pequeño con las features disponibles para crear un espacio latente musical.",
    requirement: () => Boolean(state.selectedProject?.features?.path),
    complete: () => false,
    actionLabel: "Entrenar VAE",
    renderFields: () => `
      <details class="advanced">
        <summary>Parámetros del VAE</summary>
        <div class="inline-fields">
          <label class="field">
            <span>Dimensión latente</span>
            <input id="latentDim" type="number" value="32" min="2" />
            <small>Tamaño del vector musical aprendido.</small>
          </label>
          <label class="field">
            <span>Épocas</span>
            <input id="vaeEpochs" type="number" value="200" min="1" />
            <small>Más épocas ajustan mejor, pero tardan más.</small>
          </label>
        </div>
      </details>
    `,
    run: () =>
      postJob("/api/jobs/train-vae", {
        latent_dim: toInt("#latentDim", 32),
        hidden_dim: 128,
        epochs: toInt("#vaeEpochs", 200),
        learning_rate: 0.001,
        beta: 0.001,
        seed: 42,
      }),
  },
  {
    id: "embedding",
    title: "Crear embedding del proyecto",
    help: "Codifica este proyecto con el VAE entrenado y guarda su vector latente para futuras fusiones.",
    requirement: () => Boolean(state.selectedProject?.features?.path),
    complete: () => Boolean(state.selectedProject?.embeddings?.feature_vae?.path),
    actionLabel: "Crear embedding",
    renderFields: () => "",
    run: () =>
      postJob("/api/jobs/encode-project", {
        project_id: state.selectedProjectId,
      }),
  },
  {
    id: "pretrained-generate",
    title: "Generar con modelo pre-entrenado",
    help: "Genera audio para el banco MusicCaps con uno de los 4 modelos (MusicGen-small/medium, AudioLDM2, Stable Audio Open). Conecta la GPU de Colab para que sea rápido; si no, corre en esta máquina.",
    requirement: () => true,
    complete: () => false,
    actionLabel: "Generar audio",
    renderFields: () => renderPretrainedStep(),
    run: () => {
      const engine = document.querySelector('input[name="genEngine"]:checked')?.value;
      if (!engine || engine === "custom") throw new Error("Selecciona uno de los modelos.");
      return generatePretrainedModel(engine);
    },
  },
];

function splitCsv(value) {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function parseGenreWeights(rawValue, availableGenres) {
  let payload = {};
  const raw = String(rawValue || "").trim();
  if (raw) {
    try {
      payload = JSON.parse(raw);
    } catch (error) {
      throw new Error("Los pesos deben escribirse como JSON. Ejemplo: {\"electronic\": 0.5, \"reggaeton\": 0.5}");
    }
  } else {
    const defaults = (availableGenres || []).slice(0, 2);
    payload = Object.fromEntries(defaults.map((genre) => [genre, 1]));
  }
  const weights = {};
  for (const [genre, weight] of Object.entries(payload)) {
    const numeric = Number.parseFloat(weight);
    if (!Number.isFinite(numeric) || numeric <= 0) {
      throw new Error(`Peso inválido para ${genre}. Usa números mayores a cero.`);
    }
    weights[String(genre)] = numeric;
  }
  return weights;
}

function toInt(selector, fallback) {
  const value = Number.parseInt(document.querySelector(selector)?.value || "", 10);
  return Number.isFinite(value) ? value : fallback;
}

function toFloat(selector, fallback) {
  const value = Number.parseFloat(document.querySelector(selector)?.value || "");
  return Number.isFinite(value) ? value : fallback;
}

function optionalInt(selector) {
  const raw = valueOf(selector);
  if (!raw) return null;
  const value = Number.parseInt(raw, 10);
  return Number.isFinite(value) ? value : null;
}

function optionalFloat(selector) {
  const raw = valueOf(selector);
  if (!raw) return null;
  const value = Number.parseFloat(raw);
  return Number.isFinite(value) ? value : null;
}

function checked(selector) {
  return Boolean(document.querySelector(selector)?.checked);
}

function selectedOptions(selector) {
  const element = document.querySelector(selector);
  if (!element) return [];
  return [...element.selectedOptions].map((option) => option.value).filter(Boolean);
}

function renderJamendoStorageGuide() {
  const selectedCatalog = localStorage.getItem("hybrid_selected_jamendo_catalog_path") || "Pendiente";
  const clipsCatalog = localStorage.getItem("hybrid_jamendo_clips_catalog_path") || "Se creará al cortar clips";
  const tokenManifest =
    localStorage.getItem("hybrid_jamendo_token_manifest_path") || "Se creará al procesar clips";
  const modelPath =
    localStorage.getItem("hybrid_jamendo_model_path") || "Se creará al entrenar modelo";
  const rankingPath =
    localStorage.getItem("hybrid_jamendo_ranking_path") || "Se creará al generar música";
  return `
    <div class="storage-guide">
      <strong>Dónde se guarda</strong>
      <dl>
        <div><dt>Catálogo seleccionado</dt><dd>${escapeHtml(selectedCatalog)}</dd></div>
        <div><dt>Clips WAV</dt><dd>${escapeHtml(clipsCatalog)}</dd></div>
        <div><dt>Tokens de entrenamiento</dt><dd>${escapeHtml(tokenManifest)}</dd></div>
        <div><dt>Modelo entrenado</dt><dd>${escapeHtml(modelPath)}</dd></div>
        <div><dt>Ranking / generación</dt><dd>${escapeHtml(rankingPath)}</dd></div>
      </dl>
    </div>
  `;
}

async function getJson(path) {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) throw new Error(await responseError(response));
  return response.json();
}

async function postJson(path, body) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await responseError(response));
  return response.json();
}

async function responseError(response) {
  const text = await response.text();
  try {
    const parsed = JSON.parse(text);
    return parsed.detail || text;
  } catch {
    return text || response.statusText;
  }
}

async function postJob(path, body) {
  const job = await postJson(path, body);
  state.activeJobId = job.job_id;
  state.busy = true;
  render();
  pollJob(job.job_id);
  return job;
}

async function postJobAndWait(path, body) {
  clearInterval(state.polling);
  const created = await postJson(path, body);
  state.activeJobId = created.job_id;
  state.busy = true;
  renderBusyState();
  while (true) {
    const job = await getJson(`/api/jobs/${created.job_id}`);
    renderJob(job);
    if (job.status === "completed") return job;
    if (["failed", "cancelled"].includes(job.status)) {
      throw new Error(job.message || `Job ${job.status}`);
    }
    await sleep(1200);
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function buildGenreCatalog() {
  const payload = {
    source_dir: valueOf("#genreSourceDir"),
    genres: splitCsv(valueOf("#genreNames")),
    clips_per_genre: toInt("#clipsPerGenre", 200),
    max_duration_seconds: Number.parseFloat(valueOf("#maxClipDuration") || "10"),
    catalog_name: "genre_catalog",
    source_label: "source_3",
  };
  if (!payload.source_dir) throw new Error("Indica la carpeta fuente de MIDI.");
  if (payload.genres.length < 1) throw new Error("Indica al menos un género.");
  const result = await postJson("/api/datasets/genre-catalog", payload);
  localStorage.setItem("hybrid_last_catalog_path", result.path);
  renderJson(els.datasetJson, result);
}

async function cleanMidiDataset() {
  const sourceDir = valueOf("#cleanMidiSourceDir");
  if (!sourceDir) throw new Error("Indica la carpeta MIDI a limpiar.");
  const result = await postJson("/api/datasets/clean-midi", {
    source_dir: sourceDir,
    output_name: valueOf("#cleanMidiName") || "clean_midis",
    min_duration_seconds: 1,
    max_duration_seconds: 240,
    min_notes: 4,
    min_quality_score: 0.05,
    deduplicate: true,
  });
  renderJson(els.datasetJson, result);
}

async function augmentMidiDataset() {
  const catalogPath = localStorage.getItem("hybrid_last_catalog_path") || valueOf("#genreCatalogPath");
  const sourceDir = valueOf("#genreSourceDir");
  if (!catalogPath && !sourceDir) throw new Error("Indica catálogo MIDI o carpeta fuente para augmentar.");
  const result = await postJob("/api/jobs/augment-midi", {
    catalog_path: catalogPath || null,
    source_dir: catalogPath ? null : sourceDir,
    output_name: "augmented_midis",
    transpose_steps: [-2, 0, 2],
    velocity_jitter: 8,
    timing_jitter_ticks: 12,
    quantize_step_ticks: null,
    tempo_scale: 1.0,
    seed: 42,
  });
  renderJson(els.datasetJson, result);
}

async function tokenizeCatalog() {
  const catalogPath = localStorage.getItem("hybrid_last_catalog_path");
  if (!catalogPath) throw new Error("Primero crea un catálogo JSON.");
  const result = await postJson("/api/tokens/input", {
    catalog_path: catalogPath,
    token_set_name: "genre_input_tokens",
  });
  localStorage.setItem("hybrid_jamendo_token_manifest_path", result.path);
  renderJson(els.datasetJson, result);
}

async function exportInputTokens() {
  const tokenManifestPath =
    valueOf("#exportInputTokenSelect") ||
    valueOf("#stepTokenManifestPath") ||
    localStorage.getItem("hybrid_jamendo_token_manifest_path");
  if (!tokenManifestPath) throw new Error("Primero procesa clips o tokeniza un catálogo para crear el manifest de tokens.");
  const result = await postJson("/api/tokens/input/export", {
    token_manifest_path: tokenManifestPath,
    export_name: "input_tokens_by_genre",
  });
  renderJson(els.datasetJson, result);
  triggerDownload(result.download_url);
}

async function downloadJamendo(downloadAudio) {
  let genreTags;
  try {
    genreTags = JSON.parse(valueOf("#jamendoGenreTags"));
  } catch {
    throw new Error("El JSON de géneros/tags de Jamendo no es válido.");
  }
  const maxTracksRaw = valueOf("#jamendoMaxTracks");
  const payload = {
    genre_tags: genreTags,
    catalog_name: "mtg_jamendo",
    tracks_per_page: toInt("#jamendoTracksPerPage", 200),
    max_tracks_per_genre: maxTracksRaw ? Number.parseInt(maxTracksRaw, 10) : null,
    download_audio: downloadAudio,
    client_id: "b6747d04",
    source: "mtg-cdn",
    concurrent_downloads: 8,
  };
  const result = await postJob("/api/jobs/download-jamendo", payload);
  renderJson(els.jamendoJson, result);
}

async function prepareJamendoClips() {
  const catalogPath = valueOf("#jamendoCatalogPath");
  if (!catalogPath) throw new Error("Indica la ruta del catalog.json de Jamendo.");
  const hopRaw = valueOf("#jamendoHopDuration");
  const maxRaw = valueOf("#jamendoMaxClipsPerTrack");
  const result = await postJob("/api/jobs/prepare-jamendo-clips", {
    catalog_path: catalogPath,
    clip_duration_seconds: Number.parseFloat(valueOf("#jamendoClipDuration") || "20"),
    hop_duration_seconds: hopRaw ? Number.parseFloat(hopRaw) : null,
    max_clips_per_track: maxRaw ? Number.parseInt(maxRaw, 10) : null,
    min_clip_seconds: 5,
    sample_rate: null,
    mono: true,
  });
  renderJson(els.jamendoJson, result);
}

async function processJamendoClips() {
  const clipsCatalogPath = valueOf("#clipsCatalogPath");
  if (!clipsCatalogPath) throw new Error("Indica la ruta del clips_catalog.json.");
  if (!checked("#processTrainingMaterial")) throw new Error("Para entrenar modelos necesitas crear MIDI, features y tokens.");
  const result = await postJob("/api/jobs/process-jamendo-clips", {
    clips_catalog_path: clipsCatalogPath,
    max_clips: optionalInt("#processMaxClips"),
    run_stems: checked("#processStems"),
    run_melodic: true,
    run_drums: true,
    run_features: true,
    run_tokens: true,
    continue_on_error: true,
  });
  renderJson(els.jamendoJson, result);
}

async function processJamendoClipsForTokenVae() {
  const clipsCatalogPath = valueOf("#clipsCatalogPath") || localStorage.getItem("hybrid_jamendo_clips_catalog_path");
  if (!clipsCatalogPath) throw new Error("Primero corta clips WAV para crear el clips_catalog.json.");
  const result = await postJob("/api/jobs/process-jamendo-clips", {
    clips_catalog_path: clipsCatalogPath,
    max_clips: optionalInt("#processMaxClips"),
    run_stems: true,
    run_melodic: true,
    run_drums: true,
    run_features: true,
    run_tokens: true,
    continue_on_error: true,
    processing_mode: "token_vae_demucs",
    midi_cleanup: true,
    quantize_grid: "1/16",
    strict_demucs: true,
  });
  renderJson(els.jamendoJson, result);
}

async function selectJamendoCatalog() {
  const catalogPath = valueOf("#jamendoCatalogSelect");
  if (!catalogPath) throw new Error("Selecciona un catálogo Jamendo descargado.");
  const genres = selectedOptions("#jamendoGenreSelect");
  if (!genres.length) throw new Error("Selecciona al menos un género.");
  const result = await postJson("/api/datasets/jamendo/select", {
    catalog_path: catalogPath,
    genres,
    max_tracks_per_genre: toInt("#jamendoTrackLimit", 100),
    output_name: `selected_${genres.join("_")}`,
  });
  localStorage.setItem("hybrid_selected_jamendo_catalog_path", result.path);
  const input = document.querySelector("#jamendoCatalogPath");
  if (input) input.value = result.path;
  renderJson(els.jamendoJson, result);
  render();
}

async function trainTokenModel() {
  const tokenManifestPath = valueOf("#tokenManifestSelect");
  if (!tokenManifestPath) throw new Error("Selecciona un manifest de tokens.");
  const result = await postJob("/api/jobs/train-token-model", {
    token_manifest_path: tokenManifestPath,
    model_name: valueOf("#tokenModelName") || "token_markov",
    order: toInt("#tokenModelOrder", 2),
    model_type: valueOf("#tokenModelType") || "transformer",
    sequence_length: toInt("#transformerSequenceLength", 128),
    epochs: toInt("#transformerEpochs", 8),
    batch_size: toInt("#transformerBatchSize", 16),
    embedding_dim: toInt("#transformerEmbeddingDim", 128),
    num_layers: toInt("#transformerLayers", 3),
    num_heads: toInt("#transformerHeads", 4),
  });
  renderJson(els.modelJson, result);
}

async function trainJamendoModel() {
  const tokenManifestPath =
    valueOf("#stepTokenManifestPath") || localStorage.getItem("hybrid_jamendo_token_manifest_path");
  if (!tokenManifestPath) throw new Error("Primero procesa clips para crear tokens de entrenamiento.");
  const result = await postJob("/api/jobs/train-token-model", {
    token_manifest_path: tokenManifestPath,
    model_name: valueOf("#stepTokenModelName") || "jamendo_transformer",
    order: 2,
    model_type: "transformer",
    sequence_length: toInt("#stepTransformerSequenceLength", 128),
    epochs: toInt("#stepTransformerEpochs", 8),
    batch_size: 16,
    embedding_dim: 128,
    num_layers: 3,
    num_heads: 4,
  });
  renderJson(els.modelJson, result);
}

async function generatePretrainedModel(engineOverride) {
  const engine = engineOverride || document.querySelector('input[name="genEngine"]:checked')?.value;
  if (!engine || engine === "custom") throw new Error("Selecciona un modelo pre-entrenado primero.");
  const base = cloudBase();
  if (base) return generatePretrainedCloud(base, engine);
  const result = await postJob("/api/jobs/generate-pretrained", { model_name: engine });
  renderJson(els.modelJson, result);
}

// Comprueba que el backend de nube responde y guarda su URL.
async function connectCloud() {
  const base = (valueOf("#cloudBaseInput") || "").replace(/\/+$/, "");
  const statusEl = document.querySelector("#cloudStatus");
  const dot = document.querySelector("#cloudDot");
  if (!base) {
    localStorage.removeItem("hybrid_cloud_base");
    if (statusEl) statusEl.textContent = "Sin conectar — la generación correrá localmente.";
    if (dot) dot.className = "cloud-dot";
    return;
  }
  if (statusEl) statusEl.textContent = "Conectando…";
  try {
    const r = await fetch(`${base}/api/health`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const info = await r.json();
    localStorage.setItem("hybrid_cloud_base", base);
    if (statusEl) statusEl.textContent = `Conectado · GPU: ${info.gpu || "ok"}`;
    if (dot) dot.className = "cloud-dot is-on";
  } catch (e) {
    if (statusEl) statusEl.textContent = `No se pudo conectar: ${e.message}`;
    if (dot) dot.className = "cloud-dot is-off";
  }
}

// Crea y sondea el job de generación en el backend de nube (mismo contrato
// que el backend local: status/progress/stage/message/payload).
async function generatePretrainedCloud(base, engine) {
  const dl = document.querySelector("#cloudDownload");
  if (dl) dl.innerHTML = "";
  const created = await fetch(`${base}/api/jobs/generate-pretrained`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_name: engine }),
  }).then((r) => r.json());
  state.busy = true;
  renderBusyState();
  clearInterval(state.polling);
  const tick = async () => {
    let job;
    try {
      job = await fetch(`${base}/api/jobs/${created.job_id}`).then((r) => r.json());
    } catch (e) {
      els.jobStatus.textContent = "Error";
      els.jobMessage.textContent = e.message;
      clearInterval(state.polling);
      state.busy = false;
      renderBusyState();
      return;
    }
    renderJob(job);
    if (["completed", "failed", "cancelled"].includes(job.status)) {
      clearInterval(state.polling);
      state.busy = false;
      renderBusyState();
      if (job.status === "completed" && job.payload?.download_url && dl) {
        const url = `${base}${job.payload.download_url}`;
        dl.innerHTML = `<a class="secondary-button" href="${url}" download>Descargar ${escapeHtml(engine)}.zip (${job.payload.clips ?? "?"} clips)</a>`;
      }
      renderJson(els.modelJson, job.payload || job);
    }
  };
  tick();
  state.polling = setInterval(tick, 2000);
}

async function trainTokenVae() {
  const tokenManifestPath =
    valueOf("#tokenVaeManifestSelect") ||
    valueOf("#stepTokenManifestPath") ||
    localStorage.getItem("hybrid_jamendo_token_manifest_path");
  if (!tokenManifestPath) throw new Error("Primero procesa clips o tokeniza un catálogo.");
  const result = await postJob("/api/jobs/train-token-vae", {
    token_manifest_path: tokenManifestPath,
    latent_dim: toInt("#stepTokenVaeLatentDim", 32),
    hidden_dim: 128,
    epochs: toInt("#stepTokenVaeEpochs", 60),
    learning_rate: 0.001,
    beta: 0.001,
    seed: 42,
  });
  renderJson(els.modelJson, result);
}

async function encodeTokenVae() {
  const tokenManifestPath =
    valueOf("#tokenVaeManifestSelect") ||
    valueOf("#stepTokenManifestPath") ||
    localStorage.getItem("hybrid_jamendo_token_manifest_path");
  if (!tokenManifestPath) throw new Error("Primero entrena o selecciona tokens para codificar.");
  const result = await postJob("/api/jobs/encode-token-vae", {
    token_source_path: tokenManifestPath,
    model_path: null,
    output_name: "jamendo_token_embedding",
  });
  renderJson(els.modelJson, result);
}

async function encodeGenreEmbeddings() {
  const tokenManifestPath =
    valueOf("#tokenVaeManifestSelect") ||
    valueOf("#stepTokenManifestPath") ||
    localStorage.getItem("hybrid_jamendo_token_manifest_path");
  if (!tokenManifestPath) throw new Error("Primero procesa clips o tokeniza un catálogo.");
  const result = await postJob("/api/jobs/encode-genre-embeddings", {
    token_manifest_path: tokenManifestPath,
    model_path: null,
    output_name: "jamendo_genre_embeddings",
  });
  renderJson(els.modelJson, result);
}

async function generateTokens() {
  const modelPath = valueOf("#tokenModelSelect");
  if (!modelPath) throw new Error("Selecciona un modelo token entrenado.");
  const result = await postJob("/api/jobs/generate-tokens", {
    model_path: modelPath,
    duration_seconds: toFloat("#tokenGenerationDuration", 30),
    output_name: valueOf("#generationName") || "generated_track",
    seed: optionalInt("#tokenGenerationSeed"),
    max_tokens: optionalInt("#tokenGenerationMaxTokens"),
    temperature: toFloat("#tokenGenerationTemperature", 0.9),
    top_k: optionalInt("#tokenGenerationTopK") ?? 50,
    top_p: optionalFloat("#tokenGenerationTopP") ?? 0.95,
    condition_genre: valueOf("#tokenGenerationGenre") || null,
    feature_tokens: splitCsv(valueOf("#tokenGenerationFeatures")),
    embedding_path: valueOf("#tokenGenerationEmbedding") || null,
    token_vae_embedding_path: valueOf("#tokenVaeEmbeddingSelect") || null,
    export_layers: checked("#exportLayerMidis"),
  });
  renderJson(els.modelJson, result);
}

async function generateJamendoRanked() {
  const modelPath = valueOf("#stepTokenModelPath") || localStorage.getItem("hybrid_jamendo_model_path");
  if (!modelPath) throw new Error("Primero entrena un modelo.");
  const result = await postJob("/api/jobs/generate-ranked", {
    model_path: modelPath,
    duration_seconds: toFloat("#stepGenerationDuration", 30),
    output_name: valueOf("#stepGenerationName") || "jamendo_generation",
    candidates: toInt("#stepRankedCandidates", 6),
    seed: optionalInt("#stepGenerationSeed"),
    max_tokens: optionalInt("#stepGenerationMaxTokens"),
    temperature: toFloat("#stepGenerationTemperature", 0.84),
    top_k: optionalInt("#stepGenerationTopK") ?? 56,
    top_p: optionalFloat("#stepGenerationTopP") ?? 0.92,
    condition_genre: valueOf("#stepGenerationGenre") || null,
    feature_tokens: [],
    embedding_path: null,
    token_vae_embedding_path: valueOf("#tokenVaeEmbeddingSelect") || null,
    export_layers: checked("#stepExportLayerMidis"),
    render_best: checked("#stepRankedRenderBest"),
    render_engine: valueOf("#renderEngine") || "auto",
    soundfont_path: valueOf("#renderSoundfont") || null,
    export_mp3: checked("#renderMp3"),
  });
  renderJson(els.modelJson, result);
}

async function generateRanked() {
  const modelPath = valueOf("#tokenModelSelect");
  if (!modelPath) throw new Error("Selecciona un modelo token entrenado.");
  const soundfont = valueOf("#renderSoundfont");
  const result = await postJob("/api/jobs/generate-ranked", {
    model_path: modelPath,
    duration_seconds: toFloat("#tokenGenerationDuration", 30),
    output_name: valueOf("#generationName") || "ranked_generation",
    candidates: toInt("#rankedCandidates", 6),
    seed: optionalInt("#tokenGenerationSeed"),
    max_tokens: optionalInt("#tokenGenerationMaxTokens"),
    temperature: toFloat("#tokenGenerationTemperature", 0.9),
    top_k: optionalInt("#tokenGenerationTopK") ?? 50,
    top_p: optionalFloat("#tokenGenerationTopP") ?? 0.95,
    condition_genre: valueOf("#tokenGenerationGenre") || null,
    feature_tokens: splitCsv(valueOf("#tokenGenerationFeatures")),
    embedding_path: valueOf("#tokenGenerationEmbedding") || null,
    token_vae_embedding_path: valueOf("#tokenVaeEmbeddingSelect") || null,
    export_layers: checked("#exportLayerMidis"),
    render_best: checked("#rankedRenderBest"),
    render_engine: valueOf("#renderEngine") || "auto",
    soundfont_path: soundfont || null,
    export_mp3: checked("#renderMp3"),
  });
  renderJson(els.modelJson, result);
}

function selectedGeneration() {
  const midiPath = valueOf("#generationSelect");
  return (state.resources.generations || []).find((item) => item.midi_path === midiPath) || null;
}

async function renderMidi() {
  const midiPath = valueOf("#generationSelect");
  if (!midiPath) throw new Error("Selecciona un MIDI generado.");
  const soundfont = valueOf("#renderSoundfont");
  const result = await postJob("/api/jobs/render-midi", {
    midi_path: midiPath,
    output_name: valueOf("#renderName") || "render_track",
    engine: valueOf("#renderEngine") || "auto",
    soundfont_path: soundfont || null,
    sample_rate: 44100,
    export_mp3: checked("#renderMp3"),
    pedalboard_preset: valueOf("#pedalboardPreset") || "master",
    plugin_paths: splitCsv(valueOf("#pedalboardPlugins")),
  });
  renderJson(els.renderJson, result);
}

async function renderLayers() {
  const generation = selectedGeneration();
  if (!generation?.path) throw new Error("Selecciona una generación con tokens.json disponible.");
  const soundfont = valueOf("#renderSoundfont");
  const result = await postJob("/api/jobs/render-layers", {
    generation_path: generation.path,
    output_name: `${valueOf("#renderName") || "render_track"}_layers`,
    engine: valueOf("#renderEngine") || "auto",
    soundfont_path: soundfont || null,
    sample_rate: 44100,
    export_mp3: checked("#renderMp3"),
    pedalboard_preset: valueOf("#pedalboardPreset") || "master",
    plugin_paths: splitCsv(valueOf("#pedalboardPlugins")),
  });
  renderJson(els.renderJson, result);
}

async function analyzeMidi() {
  const midiPath = valueOf("#generationSelect");
  if (!midiPath) throw new Error("Selecciona un MIDI generado.");
  const result = await postJson("/api/metrics/midi", { midi_path: midiPath });
  renderJson(els.renderJson, result);
}

async function blendEmbeddings() {
  const embeddingA = valueOf("#embeddingAPath");
  const embeddingB = valueOf("#embeddingBPath");
  if (!embeddingA || !embeddingB) throw new Error("Indica los dos embeddings a fusionar.");
  const result = await postJob("/api/jobs/blend-embeddings", {
    embedding_a_path: embeddingA,
    embedding_b_path: embeddingB,
    alpha: toFloat("#blendAlpha", 0.5),
    output_name: valueOf("#blendName") || "latent_blend",
  });
  renderJson(els.blendJson, result);
}

async function blendGenreEmbeddings() {
  const payload = genreBlendPayload();
  const result = await postJob("/api/jobs/blend-weighted-embeddings", payload);
  renderJson(els.blendJson, result);
}

function genreBlendPayload() {
  const runPath = valueOf("#genreEmbeddingRunSelect");
  if (!runPath) throw new Error("Primero crea embeddings por género.");
  const run = (state.resources.token_vae?.genre_embeddings || []).find((item) => item.path === runPath);
  if (!run) throw new Error("No se encontró el resumen de embeddings por género seleccionado.");
  const weights = parseGenreWeights(valueOf("#genreFusionWeights"), run.genres || []);
  const available = new Map((run.embeddings || []).map((item) => [String(item.genre), item]));
  const embeddings = Object.entries(weights).map(([genre, weight]) => {
    const item = available.get(genre);
    if (!item?.path) throw new Error(`No hay embedding disponible para el género: ${genre}`);
    return { path: item.path, weight, label: genre };
  });
  if (embeddings.length < 2) throw new Error("Selecciona al menos dos géneros para fusionar.");
  return {
    embeddings,
    output_name: valueOf("#genreFusionName") || "fusion_generos",
  };
}

async function generateWithGenreFusion() {
  const modelPath = valueOf("#stepTokenModelPath") || localStorage.getItem("hybrid_jamendo_model_path");
  if (!modelPath) throw new Error("Primero entrena un modelo.");
  const blendPayload = genreBlendPayload();
  const generationConfig = {
    model_path: modelPath,
    duration_seconds: toFloat("#stepGenerationDuration", 30),
    output_name: `${valueOf("#stepGenerationName") || "jamendo_generation"}_${blendPayload.output_name}`,
    candidates: toInt("#stepRankedCandidates", 6),
    seed: optionalInt("#stepGenerationSeed"),
    max_tokens: optionalInt("#stepGenerationMaxTokens"),
    temperature: toFloat("#stepGenerationTemperature", 0.84),
    top_k: optionalInt("#stepGenerationTopK") ?? 56,
    top_p: optionalFloat("#stepGenerationTopP") ?? 0.92,
    condition_genre: null,
    feature_tokens: ["fusion:direct"],
    export_layers: checked("#stepExportLayerMidis"),
    render_best: true,
    render_engine: valueOf("#renderEngine") || "auto",
    soundfont_path: valueOf("#renderSoundfont") || null,
    export_mp3: checked("#renderMp3"),
  };
  try {
    const blendJob = await postJobAndWait("/api/jobs/blend-weighted-embeddings", blendPayload);
    renderJson(els.blendJson, blendJob.result);
    const embeddingPath = blendJob.result?.blend?.path;
    if (!embeddingPath) throw new Error("La fusión terminó, pero no devolvió la ruta del embedding.");
    const generationJob = await postJobAndWait("/api/jobs/generate-ranked", {
      ...generationConfig,
      embedding_path: embeddingPath,
      token_vae_embedding_path: embeddingPath,
    });
    renderJson(els.modelJson, generationJob.result);
    state.busy = false;
    await refreshAll();
  } catch (error) {
    state.busy = false;
    renderBusyState();
    throw error;
  }
}

async function compareFusions() {
  const modelPath = valueOf("#stepTokenModelPath") || localStorage.getItem("hybrid_jamendo_model_path");
  if (!modelPath) throw new Error("Primero entrena un modelo.");
  const blends = (state.resources.blends || []).filter((blend) => blend.blend_type === "genre_fusion");
  if (!blends.length) throw new Error("Primero crea al menos una fusión de géneros.");
  const result = await postJob("/api/jobs/compare-fusions", {
    model_path: modelPath,
    fusion_embeddings: blends.map((blend) => ({
      embedding_path: blend.path,
      label: blend.label || blend.blend_id,
    })),
    duration_seconds: toFloat("#stepGenerationDuration", 30),
    output_name: `${valueOf("#stepGenerationName") || "jamendo_generation"}_fusion_compare`,
    candidates_per_fusion: Math.max(1, Math.min(toInt("#stepRankedCandidates", 3), 4)),
    seed: optionalInt("#stepGenerationSeed") ?? 42,
    max_tokens: optionalInt("#stepGenerationMaxTokens") ?? 1200,
    temperature: toFloat("#stepGenerationTemperature", 0.84),
    top_k: optionalInt("#stepGenerationTopK") ?? 56,
    top_p: optionalFloat("#stepGenerationTopP") ?? 0.92,
    feature_tokens: ["fusion:compare"],
    export_layers: checked("#stepExportLayerMidis"),
    render_best: checked("#stepRankedRenderBest"),
    render_engine: valueOf("#renderEngine") || "auto",
    soundfont_path: valueOf("#renderSoundfont") || null,
    export_mp3: checked("#renderMp3"),
  });
  renderJson(els.blendJson, result);
}

async function exportOutputTokens() {
  const selectedGenerationPath = valueOf("#exportOutputTokenSelect");
  const sourceDir = selectedGenerationPath ? parentPath(selectedGenerationPath) : latestGeneratedOutputDir();
  if (!sourceDir) throw new Error("Indica la carpeta con tokens o MIDI de salida.");
  const result = await postJson("/api/tokens/output", {
    source_dir: sourceDir,
    export_name: "mixed_output_tokens",
    duration_seconds: 30,
  });
  renderJson(els.outputJson, result);
  triggerDownload(result.download_url);
}

async function refreshAll() {
  await checkHealth();
  const projects = await getJson("/api/projects");
  state.projects = projects.projects || [];
  state.resources = await getJson("/api/resources");
  state.presets = await getJson("/api/presets");
  await fetchEvaluationState();
  if (!state.selectedProjectId && state.projects.length > 0) {
    state.selectedProjectId = state.projects[0].project_id;
  }
  await refreshSelectedProject();
  render();
}

function valueOf(selector) {
  return document.querySelector(selector)?.value?.trim() || "";
}

function latestGeneratedOutputDir() {
  const stored = localStorage.getItem("hybrid_jamendo_generation_path");
  if (stored) return parentPath(stored);
  const latest = (state.resources.generations || []).slice(-1)[0];
  if (latest?.path) return parentPath(latest.path);
  if (latest?.midi_path) return parentPath(latest.midi_path);
  return "";
}

function parentPath(path) {
  const normalized = String(path || "").replaceAll("\\", "/");
  const index = normalized.lastIndexOf("/");
  return index > 0 ? normalized.slice(0, index) : normalized;
}

function triggerDownload(url) {
  if (!url) return;
  const anchor = document.createElement("a");
  anchor.href = `${API_BASE}${encodeURI(url)}`;
  anchor.download = "";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}

function renderJson(element, payload) {
  const withAbsoluteDownload = { ...payload };
  if (withAbsoluteDownload.download_url) {
    withAbsoluteDownload.download_url = `${API_BASE}${withAbsoluteDownload.download_url}`;
  }
  const catalogPath = withAbsoluteDownload?.catalog?.path;
  if (catalogPath) {
    const input = document.querySelector("#jamendoCatalogPath");
    if (input) input.value = catalogPath;
    localStorage.setItem("hybrid_selected_jamendo_catalog_path", catalogPath);
  }
  const clipsPath = withAbsoluteDownload?.clip_catalog?.path || withAbsoluteDownload?.result?.clip_catalog?.path;
  if (clipsPath) {
    const input = document.querySelector("#clipsCatalogPath");
    if (input) input.value = clipsPath;
    localStorage.setItem("hybrid_jamendo_clips_catalog_path", clipsPath);
  }
  const tokenManifestPath =
    withAbsoluteDownload?.batch?.token_manifest_path ||
    withAbsoluteDownload?.result?.batch?.token_manifest_path ||
    (withAbsoluteDownload?.kind === "input" ? withAbsoluteDownload?.path : null);
  if (tokenManifestPath) {
    localStorage.setItem("hybrid_jamendo_token_manifest_path", tokenManifestPath);
  }
  const modelPath = withAbsoluteDownload?.model?.path || withAbsoluteDownload?.result?.model?.path;
  if (modelPath) {
    localStorage.setItem("hybrid_jamendo_model_path", modelPath);
  }
  const rankingPath = withAbsoluteDownload?.ranking?.path || withAbsoluteDownload?.result?.ranking?.path;
  if (rankingPath) {
    localStorage.setItem("hybrid_jamendo_ranking_path", rankingPath);
  }
  const generationPath =
    withAbsoluteDownload?.generation?.path || withAbsoluteDownload?.result?.generation?.path;
  if (generationPath) {
    localStorage.setItem("hybrid_jamendo_generation_path", generationPath);
  }
  if (!element) return;
  element.textContent = JSON.stringify(withAbsoluteDownload, null, 2);
}

async function checkHealth() {
  try {
    const health = await getJson("/api/health");
    els.apiStatus.innerHTML = `
      <span>Backend</span>
      <strong>Conectado</strong>
      <small>${health.project_root}</small>
      <small>Jobs: ${health.job_backend || "local"}${health.require_celery ? " · Celery obligatorio" : ""}</small>
    `;
  } catch (error) {
    els.apiStatus.innerHTML = `
      <span>Backend</span>
      <strong>No conectado</strong>
      <small>${error.message}</small>
    `;
  }
}

async function refreshSelectedProject() {
  if (!state.selectedProjectId) {
    state.selectedProject = null;
    return;
  }
  state.selectedProject = await getJson(`/api/projects/${state.selectedProjectId}`);
}

async function createProject() {
  const name = els.projectName.value.trim() || "demo";
  const result = await postJson("/api/projects", { name });
  state.selectedProjectId = result.project_id;
  els.projectName.value = "";
  await refreshAll();
}

function pollJob(jobId) {
  clearInterval(state.polling);
  const tick = async () => {
    try {
      const job = await getJson(`/api/jobs/${jobId}`);
      renderJob(job);
      if (["completed", "failed", "cancelled"].includes(job.status)) {
        clearInterval(state.polling);
        state.busy = false;
        await refreshAll();
      }
    } catch (error) {
      els.jobStatus.textContent = "Error";
      els.jobMessage.textContent = error.message;
      clearInterval(state.polling);
      state.busy = false;
    }
  };
  tick();
  state.polling = setInterval(tick, 1200);
}

function render() {
  pruneStoredResourcePaths();
  renderProjectSelect();
  renderProjectMeta();
  renderNextStep();
  renderSteps();
  refreshDynamicElements();
  renderResourceControls();
  renderPresetControls();
  renderRankings();
  renderBusyState();
  renderModeVisibility();
  applyFieldHelpTitles();
  renderEvaluationSection();
}

function pruneStoredResourcePaths() {
  pruneStoredResourcePath(
    "hybrid_jamendo_token_manifest_path",
    state.resources.token_manifests,
    (item) => item.path,
  );
}

function pruneStoredResourcePath(storageKey, items, pathGetter) {
  const storedPath = localStorage.getItem(storageKey);
  if (!storedPath) return;
  const exists = (items || []).some((item) => pathGetter(item) === storedPath);
  if (!exists) localStorage.removeItem(storageKey);
}

function renderModeVisibility() {
  if (els.localMidiPanel) {
    els.localMidiPanel.hidden = isJamendoMode();
  }
}

function refreshDynamicElements() {
  els.jamendoCatalogSelect = document.querySelector("#jamendoCatalogSelect");
  els.jamendoGenreSelect = document.querySelector("#jamendoGenreSelect");
  els.jamendoCatalogSummary = document.querySelector("#jamendoCatalogSummary");
  els.jamendoJson = document.querySelector("#jamendoJson");
  els.trainingPresetSelect = document.querySelector("#trainingPresetSelect");
  els.generationPresetSelect = document.querySelector("#generationPresetSelect");
  els.tokenManifestSelect = document.querySelector("#tokenManifestSelect");
  els.tokenModelSelect = document.querySelector("#tokenModelSelect");
  els.tokenVaeEmbeddingSelect = document.querySelector("#tokenVaeEmbeddingSelect");
  els.generationSelect = document.querySelector("#generationSelect");
  els.rankingList = document.querySelector("#rankingList");
  els.downloadList = document.querySelector("#downloadList");
  els.audioPlayers = document.querySelector("#audioPlayers");
  els.datasetJson = document.querySelector("#datasetJson");
  els.outputJson = document.querySelector("#outputJson");
  els.modelJson = document.querySelector("#modelJson");
  els.renderJson = document.querySelector("#renderJson");
  els.blendJson = document.querySelector("#blendJson");
  els.cleanMidiButton = document.querySelector("#cleanMidiButton");
  els.buildCatalogButton = document.querySelector("#buildCatalogButton");
  els.tokenizeCatalogButton = document.querySelector("#tokenizeCatalogButton");
  els.exportInputTokensButton = document.querySelector("#exportInputTokensButton");
  els.exportOutputTokensButton = document.querySelector("#exportOutputTokensButton");
  els.exportInputTokenSelect = document.querySelector("#exportInputTokenSelect");
  els.exportOutputTokenSelect = document.querySelector("#exportOutputTokenSelect");
  els.trainTokenModelButton = document.querySelector("#trainTokenModelButton");
  els.generateTokensButton = document.querySelector("#generateTokensButton");
  els.generateRankedButton = document.querySelector("#generateRankedButton");
  els.renderMidiButton = document.querySelector("#renderMidiButton");
  els.renderLayersButton = document.querySelector("#renderLayersButton");
  els.analyzeMidiButton = document.querySelector("#analyzeMidiButton");
  els.blendEmbeddingsButton = document.querySelector("#blendEmbeddingsButton");
}

function renderPresetControls() {
  renderPresetSelect(els.trainingPresetSelect, state.presets.training, "Sin preset");
  renderPresetSelect(els.generationPresetSelect, state.presets.generation, "Sin preset");
}

function renderPresetSelect(element, rows, emptyLabel) {
  if (!element) return;
  const current = element.value;
  const options = [`<option value="">${escapeHtml(emptyLabel)}</option>`]
    .concat(
      Object.entries(rows || {}).map(
        ([key, preset]) =>
          `<option value="${escapeHtml(key)}" ${key === current ? "selected" : ""}>${escapeHtml(preset.label || key)}</option>`,
      ),
    )
    .join("");
  element.innerHTML = options;
  if (current && [...element.options].some((option) => option.value === current)) {
    element.value = current;
  }
}

function applyTrainingPreset() {
  const preset = state.presets.training?.[els.trainingPresetSelect.value];
  if (!preset) return;
  setValue("#tokenModelType", preset.model_type || "transformer");
  setValue("#tokenModelOrder", preset.order ?? 2);
  setValue("#transformerSequenceLength", preset.sequence_length ?? 128);
  setValue("#transformerEpochs", preset.epochs ?? 8);
  setValue("#transformerBatchSize", preset.batch_size ?? 16);
  setValue("#transformerEmbeddingDim", preset.embedding_dim ?? 128);
  setValue("#transformerLayers", preset.num_layers ?? 3);
  setValue("#transformerHeads", preset.num_heads ?? 4);
}

function applyGenerationPreset() {
  const preset = state.presets.generation?.[els.generationPresetSelect.value];
  if (!preset) return;
  setValue("#tokenGenerationDuration", preset.duration_seconds ?? 30);
  setValue("#tokenGenerationMaxTokens", preset.max_tokens ?? "");
  setValue("#stepGenerationDuration", preset.duration_seconds ?? 30);
  setValue("#stepGenerationMaxTokens", preset.max_tokens ?? "");
  setValue("#tokenGenerationTemperature", preset.temperature ?? 0.9);
  setValue("#tokenGenerationTopK", preset.top_k ?? 50);
  setValue("#tokenGenerationTopP", preset.top_p ?? 0.95);
  setValue("#stepGenerationTemperature", preset.temperature ?? 0.84);
  setValue("#stepGenerationTopK", preset.top_k ?? 56);
  setValue("#stepGenerationTopP", preset.top_p ?? 0.92);
  setValue("#tokenGenerationFeatures", (preset.feature_tokens || []).join(","));
  const layers = document.querySelector("#exportLayerMidis");
  if (layers) layers.checked = Boolean(preset.export_layers);
}

function setValue(selector, value) {
  const element = document.querySelector(selector);
  if (element) element.value = value;
}

function renderResourceControls() {
  pruneStoredResourcePath(
    "hybrid_jamendo_token_manifest_path",
    state.resources.token_manifests,
    (item) => item.path,
  );
  renderSelect(
    els.tokenManifestSelect,
    state.resources.token_manifests,
    "No hay tokens todavía",
    (item) => item.path,
    (item) => item.label,
  );
  setSelectIfAvailable(els.tokenManifestSelect, localStorage.getItem("hybrid_jamendo_token_manifest_path"));
  const inputTokenManifests = (state.resources.token_manifests || []).filter((item) => item.kind !== "output");
  const tokenVaeManifests = inputTokenManifests
    .filter((item) => item.processing_mode === "token_vae_demucs")
    .concat(inputTokenManifests.filter((item) => item.processing_mode !== "token_vae_demucs"));
  renderSelect(
    els.exportInputTokenSelect,
    tokenVaeManifests,
    "No hay tokens de entrada todavía",
    (item) => item.path,
    (item) => item.label,
  );
  setSelectIfAvailable(els.exportInputTokenSelect, localStorage.getItem("hybrid_jamendo_token_manifest_path"));
  renderSelect(
    document.querySelector("#tokenVaeManifestSelect"),
    tokenVaeManifests,
    "No hay tokens Token-VAE todavía",
    (item) => item.path,
    (item) => item.label,
  );
  setSelectIfAvailable(document.querySelector("#tokenVaeManifestSelect"), localStorage.getItem("hybrid_jamendo_token_manifest_path"));
  renderSelect(
    els.tokenModelSelect,
    state.resources.token_models,
    "No hay modelos todavía",
    (item) => item.path,
    (item) => item.label,
  );
  setSelectIfAvailable(els.tokenModelSelect, localStorage.getItem("hybrid_jamendo_model_path"));
  renderGroupedSelect(
    els.tokenVaeEmbeddingSelect,
    embeddingSelectOptions(),
    "Sin embeddings Token-VAE",
    (item) => item.path,
    (item) => item.label,
  );
  renderGroupedSelect(
    document.querySelector("#embeddingAPath"),
    embeddingSelectOptions(),
    "Selecciona embedding A",
    (item) => item.path,
    (item) => item.label,
  );
  renderGroupedSelect(
    document.querySelector("#embeddingBPath"),
    embeddingSelectOptions(),
    "Selecciona embedding B",
    (item) => item.path,
    (item) => item.label,
  );
  renderSelect(
    document.querySelector("#genreEmbeddingRunSelect"),
    state.resources.token_vae?.genre_embeddings || [],
    "Sin embeddings por género",
    (item) => item.path,
    (item) => item.label,
  );
  renderSelect(
    document.querySelector("#embeddingProjectionRun"),
    state.resources.token_vae?.genre_embeddings || [],
    "Sin embeddings por género",
    (item) => item.path,
    (item) => item.label,
  );
  renderSelect(
    els.generationSelect,
    state.resources.generations,
    "No hay MIDIs generados",
    (item) => item.midi_path,
    (item) => item.label,
  );
  renderSelect(
    els.exportOutputTokenSelect,
    state.resources.generations,
    "No hay tokens de salida todavía",
    (item) => item.path,
    (item) => item.label,
  );
  setSelectIfAvailable(els.exportOutputTokenSelect, localStorage.getItem("hybrid_jamendo_generation_path"));
  renderSelect(
    els.jamendoCatalogSelect,
    state.resources.jamendo_catalogs,
    "No hay catálogos Jamendo",
    (item) => item.path,
    (item) => `${item.catalog_name || item.catalog_id} · ${item.total_tracks || 0} pistas`,
  );
  renderJamendoGenreOptions();
  renderDownloads();
  renderAudioPlayers();
}

function embeddingSelectOptions() {
  const tokenEmbeddings = (state.resources.token_vae?.embeddings || []).map((item) => ({
    ...item,
    group: "Embeddings normales",
    label: item.label,
  }));
  const genreEmbeddings = (state.resources.token_vae?.genre_embeddings || []).flatMap((run) =>
    (run.embeddings || []).map((item) => ({
      ...item,
      group: "Embeddings por género",
      label: `${item.genre} · ${run.run_id || "embedding"}`,
    })),
  );
  const genreFusions = (state.resources.blends || [])
    .filter((item) => item.blend_type === "genre_fusion")
    .map((item) => ({
      ...item,
      group: "Fusiones de géneros",
      label: item.label,
    }));
  const pairBlends = (state.resources.blends || [])
    .filter((item) => item.blend_type !== "genre_fusion")
    .map((item) => ({
      ...item,
      group: "Fusiones A/B",
      label: item.label,
    }));
  return [...tokenEmbeddings, ...genreEmbeddings, ...genreFusions, ...pairBlends].filter((item) => item.path);
}

function renderJamendoGenreOptions() {
  if (!els.jamendoGenreSelect) return;
  if (els.jamendoCatalogSelect && !els.jamendoCatalogSelect.value) {
    const firstCatalog = (state.resources.jamendo_catalogs || [])[0];
    if (firstCatalog?.path) els.jamendoCatalogSelect.value = firstCatalog.path;
  }
  const catalog = selectedJamendoCatalog();
  if (!catalog) {
    els.jamendoGenreSelect.innerHTML = `<option value="">Sin catálogo descargado</option>`;
    if (els.jamendoCatalogSummary) {
      els.jamendoCatalogSummary.textContent = "Descarga o selecciona un catálogo Jamendo primero.";
    }
    return;
  }
  const selected = new Set([...els.jamendoGenreSelect.selectedOptions].map((option) => option.value));
  const counts = catalog.counts || {};
  const genres = Object.keys(counts).sort();
  els.jamendoGenreSelect.innerHTML = genres
    .map((genre) => {
      const count = counts[genre] || 0;
      return `<option value="${escapeHtml(genre)}" ${
        selected.has(genre) ? "selected" : ""
      }>${escapeHtml(genre)} · ${count} pistas</option>`;
    })
    .join("");
  if (els.jamendoCatalogSummary) {
    els.jamendoCatalogSummary.textContent = `${catalog.total_tracks || 0} pistas disponibles en ${
      genres.length
    } géneros. Elige uno o varios y define cuántas usar.`;
  }
}

function selectedJamendoCatalog() {
  const path = valueOf("#jamendoCatalogSelect");
  return (state.resources.jamendo_catalogs || []).find((item) => item.path === path) || null;
}

function renderSelect(element, rows, emptyLabel, valueOfItem, labelOfItem) {
  if (!element) return;
  const current = element.value;
  const options = [`<option value="">${escapeHtml(emptyLabel)}</option>`]
    .concat(
      (rows || []).map((item) => {
        const value = valueOfItem(item) || "";
        return `<option value="${escapeHtml(value)}" ${
          value === current ? "selected" : ""
        }>${escapeHtml(labelOfItem(item))}</option>`;
      }),
    )
    .join("");
  element.innerHTML = options;
  if (current && [...element.options].some((option) => option.value === current)) {
    element.value = current;
  }
}

function renderGroupedSelect(element, rows, emptyLabel, valueOfItem, labelOfItem) {
  if (!element) return;
  const current = element.value;
  const groups = new Map();
  for (const item of rows || []) {
    const group = item.group || "Otros";
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(item);
  }
  const groupedOptions = [...groups.entries()]
    .map(([group, items]) => {
      const options = items
        .map((item) => {
          const value = valueOfItem(item) || "";
          return `<option value="${escapeHtml(value)}" ${
            value === current ? "selected" : ""
          }>${escapeHtml(labelOfItem(item))}</option>`;
        })
        .join("");
      return `<optgroup label="${escapeHtml(group)}">${options}</optgroup>`;
    })
    .join("");
  element.innerHTML = `<option value="">${escapeHtml(emptyLabel)}</option>${groupedOptions}`;
  if (current && [...element.options].some((option) => option.value === current)) {
    element.value = current;
  }
}

function setSelectIfAvailable(element, value) {
  if (!element || !value) return;
  if ([...element.options].some((option) => option.value === value)) {
    element.value = value;
  }
}

function renderDownloads() {
  const links = [];
  for (const manifest of state.resources.token_manifests || []) {
    if (manifest.zip_download_url) {
      links.push({
        label: `Tokens ${manifest.kind || "manifest"} · ${manifest.total_files} archivos`,
        url: manifest.zip_download_url,
      });
    }
  }
  for (const ranking of state.resources.rankings || []) {
    if (ranking.ranking_download_url) {
      links.push({ label: `${ranking.generation_mode_label || "Generación"} · Ranking · ${ranking.ranked_id}`, url: ranking.ranking_download_url });
    }
    if (ranking.best_midi_download_url) {
      links.push({ label: `${ranking.generation_mode_label || "Generación"} · Mejor MIDI · ${ranking.ranked_id}`, url: ranking.best_midi_download_url });
    }
  }
  for (const generation of state.resources.generations || []) {
    if (generation.midi_download_url) {
      links.push({
        label: `${generation.generation_mode_label || "Generación"} · MIDI · ${generation.generation_id}`,
        url: generation.midi_download_url,
      });
    }
    if (generation.tokens_download_url) {
      links.push({
        label: `${generation.generation_mode_label || "Generación"} · Tokens · ${generation.generation_id}`,
        url: generation.tokens_download_url,
      });
    }
    for (const [layerName, layer] of Object.entries(generation.layer_midis || {})) {
      if (layer.download_url) {
        links.push({
          label: `${generation.generation_mode_label || "Generación"} · ${layerName}.mid · ${generation.generation_id}`,
          url: layer.download_url,
        });
      }
    }
  }
  for (const render of state.resources.renders || []) {
    const mode = render.generation_mode_label || "Render de audio";
    if (render.wav_download_url) links.push({ label: `${mode} · WAV · ${render.label}`, url: render.wav_download_url });
    if (render.mp3_download_url) links.push({ label: `${mode} · MP3 · ${render.label}`, url: render.mp3_download_url });
  }
  els.downloadList.innerHTML = links.length
    ? links
        .slice(-30)
        .reverse()
        .map((link) => `<a href="${API_BASE}${encodeURI(link.url)}" download>${escapeHtml(link.label)}</a>`)
        .join("")
    : `<small class="muted">Cuando generes MIDI o audio aparecerán aquí las descargas.</small>`;
}

const AUDIO_GENRE_ORDER = ["classical", "electronic", "reggaeton"];
const AUDIO_GENRE_LABELS = {
  classical: "Classical",
  electronic: "Electronic",
  reggaeton: "Reggaeton",
  _none: "Sin género",
};
const AUDIO_TOP_PER_GENRE = 5;

function detectAudioGenre(text) {
  const low = String(text || "").toLowerCase();
  for (const genre of AUDIO_GENRE_ORDER) {
    if (low.includes(genre)) return genre;
  }
  return "_none";
}

function audioPlayerCard(item, rankIndex) {
  return `
    <li class="audio-card">
      <div class="audio-card-head">
        <span class="rank-badge">#${rankIndex}</span>
        <span class="mode-badge ${escapeHtml(item.mode || "render")}">${escapeHtml(item.modeLabel || "Render")}</span>
        ${item.score != null ? `<span class="score-pill">score ${formatNumber(item.score)}</span>` : ""}
        <span class="audio-title">${escapeHtml(item.title || "")}</span>
      </div>
      <audio controls src="${API_BASE}${encodeURI(item.url)}"></audio>
      <span class="file-links">
        ${item.midiUrl ? `<a href="${API_BASE}${encodeURI(item.midiUrl)}" download>MIDI</a>` : ""}
        ${item.wavUrl ? `<a href="${API_BASE}${encodeURI(item.wavUrl)}" download>WAV</a>` : ""}
        ${item.mp3Url ? `<a href="${API_BASE}${encodeURI(item.mp3Url)}" download>MP3</a>` : ""}
      </span>
    </li>
  `;
}

function renderAudioPlayers() {
  if (!els.audioPlayers) return;
  const playable = [];
  for (const render of state.resources.renders || []) {
    const url = render.mp3_download_url || render.wav_download_url;
    if (url) {
      playable.push({
        modeLabel: render.generation_mode_label || "Render de audio",
        title: render.label,
        mode: render.generation_mode || "render",
        genre: detectAudioGenre(`${render.label} ${render.generation_mode_label || ""}`),
        score: null,
        url,
        wavUrl: render.wav_download_url,
        mp3Url: render.mp3_download_url,
        midiUrl: render.midi_download_url || null,
      });
    }
  }
  for (const ranking of state.resources.rankings || []) {
    const genre = ranking.condition_genre || detectAudioGenre(`${ranking.ranked_id} ${ranking.label}`);
    for (const candidate of ranking.candidates || []) {
      const url = candidate.mp3_download_url || candidate.wav_download_url;
      if (url) {
        playable.push({
          modeLabel: ranking.generation_mode_label || "Generación",
          title: `${ranking.ranked_id} · ${candidate.candidate_id}`,
          mode: ranking.generation_mode || "transformer",
          genre,
          score: typeof candidate.score === "number" ? candidate.score : null,
          url,
          wavUrl: candidate.wav_download_url,
          mp3Url: candidate.mp3_download_url,
          midiUrl: candidate.midi_download_url,
        });
      }
    }
  }

  if (!playable.length) {
    els.audioPlayers.innerHTML = `<small class="muted">Cuando renderices WAV/MP3 podrás escucharlos aquí.</small>`;
    return;
  }

  // Agrupar por género y mostrar SOLO el top-5 por puntaje; el resto va en un
  // desplegable "ver más" para no saturar la vista.
  const groups = {};
  for (const item of playable) {
    (groups[item.genre] = groups[item.genre] || []).push(item);
  }
  const orderedGenres = [
    ...AUDIO_GENRE_ORDER.filter((g) => groups[g]),
    ...Object.keys(groups).filter((g) => !AUDIO_GENRE_ORDER.includes(g) && g !== "_none"),
    ...(groups._none ? ["_none"] : []),
  ];

  els.audioPlayers.innerHTML = orderedGenres
    .map((genre) => {
      const items = groups[genre]
        .slice()
        .sort((a, b) => (b.score == null ? -Infinity : b.score) - (a.score == null ? -Infinity : a.score));
      const top = items.slice(0, AUDIO_TOP_PER_GENRE);
      const rest = items.slice(AUDIO_TOP_PER_GENRE);
      const label = AUDIO_GENRE_LABELS[genre] || genre;
      const topHtml = top.map((item, i) => audioPlayerCard(item, i + 1)).join("");
      const restHtml = rest.length
        ? `<details class="audio-more">
             <summary>Ver las otras ${rest.length} pista${rest.length === 1 ? "" : "s"} de ${escapeHtml(label)}</summary>
             <ol class="audio-ranking">${rest.map((item, i) => audioPlayerCard(item, AUDIO_TOP_PER_GENRE + i + 1)).join("")}</ol>
           </details>`
        : "";
      return `
        <section class="audio-genre">
          <header class="audio-genre-head">
            <h3>${escapeHtml(label)}</h3>
            <span class="muted">${items.length} pista${items.length === 1 ? "" : "s"} · top ${Math.min(AUDIO_TOP_PER_GENRE, items.length)} por puntaje</span>
          </header>
          <ol class="audio-ranking">${topHtml}</ol>
          ${restHtml}
        </section>
      `;
    })
    .join("");
}

function renderRankings() {
  if (!els.rankingList) return;
  const rankings = (state.resources.rankings || []).slice(-5).reverse();
  els.rankingList.innerHTML = rankings.length
    ? rankings
        .map(
          (ranking) => `
            <div class="ranking-card">
              <div class="ranking-row">
                <strong>${escapeHtml(ranking.best_candidate_id || "candidate-01")}</strong>
                <div>
                  <span class="mode-badge ${escapeHtml(ranking.generation_mode || "transformer")}">${escapeHtml(
                    ranking.generation_mode_label || "Transformer normal",
                  )}</span><br />
                  <small>${escapeHtml(ranking.ranked_id)}</small><br />
                  <small>${escapeHtml(ranking.created_at || "")}</small>
                </div>
                <strong>${formatNumber(ranking.best_score)}</strong>
              </div>
              ${renderRankingCandidates(ranking)}
            </div>
          `,
        )
        .join("")
    : `<small class="muted">Cuando uses “Generar y elegir mejor”, el ranking aparecerá aquí.</small>`;
}

function renderRankingCandidates(ranking) {
  const candidates = (ranking.candidates || [])
    .slice()
    .sort((a, b) => (a.rank || 99) - (b.rank || 99));
  if (!candidates.length) return "";
  return `
    <div class="candidate-table" aria-label="Comparación de candidatas">
      <div class="candidate-head">
        <span>Rank</span>
        <span>Score</span>
        <span>Seed</span>
        <span>Escuchar / descargar</span>
      </div>
      ${candidates
        .map((candidate) => {
          const audioUrl = candidate.mp3_download_url || candidate.wav_download_url;
          const midiUrl = candidate.midi_download_url;
          const wavUrl = candidate.wav_download_url;
          const mp3Url = candidate.mp3_download_url;
          const renderButton =
            !audioUrl && candidate.midi_path
              ? `<button class="quiet-button" data-render-candidate="${escapeHtml(candidate.midi_path)}" type="button">Renderizar audio</button>`
              : "";
          return `
            <div class="candidate-row">
              <span>${escapeHtml(candidate.rank ?? "")}</span>
              <span>${formatNumber(candidate.score)}</span>
              <span>${escapeHtml(candidate.seed ?? "auto")}</span>
              <span class="candidate-actions">
                ${
                  audioUrl
                    ? `<audio controls src="${API_BASE}${encodeURI(audioUrl)}"></audio>`
                    : `<small class="muted">Sin audio renderizado</small>`
                }
                <span class="file-links">
                  ${midiUrl ? `<a href="${API_BASE}${encodeURI(midiUrl)}" download>MIDI</a>` : ""}
                  ${wavUrl ? `<a href="${API_BASE}${encodeURI(wavUrl)}" download>WAV</a>` : ""}
                  ${mp3Url ? `<a href="${API_BASE}${encodeURI(mp3Url)}" download>MP3</a>` : ""}
                </span>
                ${renderButton}
              </span>
            </div>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderBusyState() {
  [
    els.cleanMidiButton,
    els.trainTokenModelButton,
    els.generateTokensButton,
    els.generateRankedButton,
    els.renderMidiButton,
    els.renderLayersButton,
    els.analyzeMidiButton,
    els.exportInputTokensButton,
    els.exportOutputTokensButton,
    els.blendEmbeddingsButton,
  ].forEach((button) => {
    if (button) button.disabled = state.busy;
  });
  document.querySelectorAll("button[data-jamendo-action]").forEach((button) => {
    button.disabled = state.busy;
  });
  document.querySelectorAll("button[data-cycle-action]").forEach((button) => {
    button.disabled = state.busy;
  });
  ["#metricsGenerateBatchButton", "#metricsRunButton", "#metricsRefreshButton", "#generatedMetricsRunButton", "#generatedMetricsRefreshButton"].forEach((selector) => {
    const button = document.querySelector(selector);
    if (button) button.disabled = state.busy;
  });
}

function applyFieldHelpTitles() {
  document.querySelectorAll(".field").forEach((field) => {
    const help = field.querySelector("small")?.textContent?.trim();
    const label = field.querySelector("span")?.textContent?.trim();
    if (!help) return;
    field.querySelectorAll("input, select, textarea").forEach((control) => {
      if (!control.getAttribute("title")) control.setAttribute("title", help);
      if (label && !control.getAttribute("aria-label")) control.setAttribute("aria-label", label);
    });
  });
}

function formatNumber(value) {
  const number = Number.parseFloat(value);
  if (!Number.isFinite(number)) return "0.000";
  return number.toFixed(3);
}

function renderProjectSelect() {
  const options = [`<option value="">Sin proyecto</option>`]
    .concat(
      state.projects.map(
        (project) =>
          `<option value="${escapeHtml(project.project_id)}" ${
            project.project_id === state.selectedProjectId ? "selected" : ""
          }>${escapeHtml(project.name)} · ${escapeHtml(project.status)}</option>`,
      ),
    )
    .join("");
  els.projectSelect.innerHTML = options;
}

function renderProjectMeta() {
  const project = state.selectedProject;
  if (!project) {
    els.projectMeta.innerHTML = `<div><dt>Estado</dt><dd>Crea o selecciona un proyecto.</dd></div>`;
    return;
  }
  els.projectMeta.innerHTML = `
    <div><dt>ID</dt><dd>${escapeHtml(project.project_id)}</dd></div>
    <div><dt>Estado</dt><dd>${escapeHtml(project.status)}</dd></div>
    <div><dt>Actualizado</dt><dd>${escapeHtml(project.updated_at || "")}</dd></div>
  `;
}

function renderNextStep() {
  const next = activeStepDefinitions().find((step) => !step.complete() && step.requirement());
  if (!state.selectedProject) {
    els.nextStep.innerHTML = `
      <span>Siguiente paso</span>
      <strong>Crea un proyecto o selecciona uno existente.</strong>
      <p>Un proyecto agrupa audio, stems, MIDI, features y embeddings.</p>
    `;
    return;
  }
  if (!next) {
    if (isJamendoMode()) {
      els.nextStep.innerHTML = `
        <span>Siguiente paso</span>
        <strong>Cierra el ciclo: escucha, descarga o reitera.</strong>
        <p>Renderiza el MIDI generado a WAV/MP3, compara candidatas y exporta tokens si quieres continuar otra iteración.</p>
      `;
      return;
    }
    els.nextStep.innerHTML = `
      <span>Siguiente paso</span>
      <strong>El proyecto ya tiene la base técnica completa.</strong>
      <p>Puedes revisar resultados o repetir algún paso con nuevos parámetros.</p>
    `;
    return;
  }
  els.nextStep.innerHTML = `
    <span>Siguiente paso</span>
    <strong>${escapeHtml(next.title)}</strong>
    <p>${escapeHtml(next.help)}</p>
  `;
}

function renderSteps() {
  const definitions = activeStepDefinitions();
  const activeIndex = definitions.findIndex((step) => !step.complete() && step.requirement());
  els.steps.innerHTML = definitions
    .map((step, index) => {
      const ready = step.requirement();
      const complete = step.complete();
      const disabled = !ready || state.busy;
      const isActive = index === activeIndex;
      const statusClass = complete ? "" : ready ? "warn" : "error";
      const statusText = complete ? "Listo" : ready ? "Pendiente" : "Bloqueado";
      return `
        <article class="step${isActive ? " is-active" : ""}" data-step-id="${escapeHtml(step.id)}">
          <div>
            <div class="step-header">
              <span class="step-number">Paso ${index + 1} · <b class="status-pill ${statusClass}">${statusText}</b>${isActive ? ` <b class="status-pill active-step-pill">Paso actual</b>` : ""}</span>
              <h2>${escapeHtml(step.title)}</h2>
              <p>${escapeHtml(step.help)}</p>
            </div>
            ${step.renderFields()}
          </div>
          <div class="step-actions">
            <button class="primary-button" data-step="${step.id}" data-action="primary" ${disabled ? "disabled" : ""}>
              ${escapeHtml(step.actionLabel)}
            </button>
            ${
              step.secondaryLabel
                ? `<button class="secondary-button" data-step="${step.id}" data-action="secondary" ${disabled ? "disabled" : ""}>${escapeHtml(step.secondaryLabel)}</button>`
                : ""
            }
          </div>
        </article>
      `;
    })
    .join("");
  focusActiveStep(activeIndex >= 0 ? definitions[activeIndex].id : "");
  applyAdvancedVisibility();
}

// Interruptor global "Mostrar opciones avanzadas": abre/cierra TODOS los
// <details class="advanced"> de la página (los pasos se re-renderizan, así que
// hay que reaplicar el estado tras cada renderSteps). Se persiste en localStorage.
function isAdvancedVisible() {
  return localStorage.getItem("hybrid_show_advanced") === "1";
}

function applyAdvancedVisibility() {
  const show = isAdvancedVisible();
  if (els.toggleAdvanced) els.toggleAdvanced.checked = show;
  document.querySelectorAll("details.advanced").forEach((node) => {
    node.open = show;
  });
}

// Desplaza la vista hacia el paso activo (el "siguiente paso") cuando este cambia,
// para que la barra de actividad y el trabajo en curso queden visibles.
function focusActiveStep(stepId) {
  if (stepId === state.lastFocusedStepId) return;
  const previous = state.lastFocusedStepId;
  state.lastFocusedStepId = stepId;
  if (!state.stepsInitialized) {
    // En la primera carga solo registramos el paso, sin mover la vista.
    state.stepsInitialized = true;
    return;
  }
  if (!stepId || !previous) return;
  scrollStepIntoView(stepId);
}

function scrollStepIntoView(stepId) {
  const target = els.steps?.querySelector(`[data-step-id="${(window.CSS && CSS.escape) ? CSS.escape(stepId) : stepId}"]`);
  if (target) {
    target.scrollIntoView({ behavior: "smooth", block: "center" });
  } else if (els.activityPanel) {
    els.activityPanel.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

function activeStepDefinitions() {
  const jamendoMode = isJamendoMode();
  return stepDefinitions.filter((step) => {
    if (step.mode === "jamendo") return jamendoMode;
    if (jamendoMode && ["stems", "midi", "features", "vae", "embedding", "pretrained-generate"].includes(step.id)) {
      return false;
    }
    return step.mode !== "jamendo";
  });
}

function isJamendoMode() {
  return Boolean(localStorage.getItem("hybrid_selected_jamendo_catalog_path"));
}

function renderJob(job) {
  els.jobStatus.textContent = job.status || "En espera";
  els.jobProgress.style.width = `${Math.round((job.progress || 0) * 100)}%`;
  els.jobMessage.textContent = `${job.stage || ""}: ${job.message || ""}`;
  // Cuando un job arranca, llevamos la vista al paso activo para que la barra
  // de progreso de actividad sea visible mientras avanza.
  const running = Boolean(job.status) && !["completed", "failed", "cancelled", "En espera"].includes(job.status);
  if (running && !state.lastJobRunning) {
    scrollStepIntoView(state.lastFocusedStepId);
  }
  state.lastJobRunning = running;
  const events = (job.events || []).slice(-8).reverse();
  els.jobEvents.innerHTML = events
    .map((event) => `<li>${escapeHtml(event.time || "")} · ${escapeHtml(event.stage || "")}<br />${escapeHtml(event.message || "")}</li>`)
    .join("");
  if (job.status === "completed" && job.result) {
    if (["train-token-model", "generate-tokens", "generate-ranked", "train-token-vae", "encode-token-vae", "encode-genre-embeddings"].includes(job.kind)) {
      renderJson(els.modelJson, job.result);
    } else if (["render-midi", "render-layers", "midi-metrics"].includes(job.kind)) {
      renderJson(els.renderJson, job.result);
    } else if (["evaluation-from-results", "evaluation-run", "evaluation-generate-batch"].includes(job.kind)) {
      renderJson(els.outputJson, job.result);
    } else if (["blend-embeddings", "blend-weighted-embeddings", "compare-fusions"].includes(job.kind)) {
      renderJson(els.blendJson, job.result);
    } else if (["download-jamendo", "prepare-jamendo-clips", "process-jamendo-clips"].includes(job.kind)) {
      renderJson(els.jamendoJson, job.result);
    }
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}



async function fetchEvaluationState() {
  try {
    const availability = await getJson("/api/evaluations/availability");
    const sources = await getJson("/api/evaluations/generated-sources");
    const evaluations = await getJson("/api/evaluations");
    state.evaluationAvailability = availability;
    state.evaluationSources = sources;
    state.evaluations = evaluations.evaluations || [];
    const generatedReports = {};
    await Promise.all(
      (state.evaluations || [])
        .filter((item) => item.report_path)
        .map(async (item) => {
          try {
            generatedReports[item.evaluation_id] = await getJson(`/api/evaluations/${encodeURIComponent(item.evaluation_id)}/report`);
          } catch (_error) {
            generatedReports[item.evaluation_id] = null;
          }
        }),
    );
    state.evaluationReports = generatedReports;
    state.evaluationError = "";
  } catch (error) {
    state.evaluationAvailability = null;
    state.evaluationSources = null;
    state.evaluations = [];
    state.evaluationReports = {};
    state.evaluationError = error.message || String(error);
  }
}

function renderEvaluationSection() {
  const section = document.querySelector("#evaluacion-real");
  if (!section) return;
  const status = document.querySelector("#metricsStatus");
  const body = document.querySelector("#metricsDistributionBody");
  const modelSelect = document.querySelector("#metricsModelSelect");
  const evaluationSelect = document.querySelector("#metricsEvaluationSelect");
  renderGeneratedEvaluationPanel();
  if (state.evaluationError) {
    if (status) status.textContent = `Evaluación no disponible: ${state.evaluationError}`;
  } else if (status && state.evaluationAvailability) {
    const counts = state.evaluationAvailability.counts || {};
    const total = Object.values(counts).reduce((sum, value) => sum + Number(value || 0), 0);
    status.textContent = `Audios reales disponibles: ${total}. Distribución recomendada dinámica para ${state.evaluationAvailability.target_total || 100} canciones.`;
  }

  if (body) {
    const counts = state.evaluationAvailability?.counts || {};
    const recommended = state.evaluationAvailability?.recommended_distribution || {};
    const genres = Object.keys(counts).sort();
    body.innerHTML = genres.length
      ? genres
          .map(
            (genre) => `
              <tr>
                <td>${escapeHtml(genre)}</td>
                <td>${Number(counts[genre] || 0)}</td>
                <td><input class="metrics-distribution-input" data-genre="${escapeHtml(genre)}" type="number" min="0" max="${Number(counts[genre] || 0)}" value="${Number(recommended[genre] || 0)}" title="Cantidad de canciones nuevas a generar para ${escapeHtml(genre)}" /></td>
              </tr>
            `,
          )
          .join("")
      : `<tr><td colspan="3">No hay audios reales organizados por género.</td></tr>`;
  }

  renderSelect(
    modelSelect,
    state.resources.token_models || [],
    "No hay modelos entrenados",
    (item) => item.path,
    (item) => item.label,
  );
  setSelectIfAvailable(modelSelect, localStorage.getItem("hybrid_jamendo_model_path"));

  renderSelect(
    evaluationSelect,
    state.evaluations || [],
    "No hay evaluaciones generadas",
    (item) => item.evaluation_id,
    (item) => `${item.evaluation_id} · ${item.status || "sin estado"}`,
  );
  renderEvaluationSummary();
}

function renderGeneratedEvaluationPanel() {
  const status = document.querySelector("#generatedMetricsStatus");
  const evaluationSelect = document.querySelector("#generatedEvaluationSelect");
  const matrix = document.querySelector("#generatedMetricsMatrix");
  const genreGroups = generatedGenreGroups();
  const currentMatrixValues = readGeneratedMatrixValues();
  if (state.evaluationError) {
    if (status) status.textContent = `Métricas no disponibles: ${state.evaluationError}`;
  } else if (status) {
    const total = state.evaluationSources?.total_audio_ready || 0;
    status.textContent = total
      ? `${total} candidatas con WAV/MP3 listas para evaluar, organizadas por género y corrida.`
      : "No hay candidatas con audio. Renderiza WAV/MP3 desde el ranking antes de calcular métricas.";
  }
  if (matrix) {
    const genres = Object.keys(genreGroups).sort();
    matrix.innerHTML = genres.length
      ? genres.map((genre) => renderGeneratedGenreMatrix(genre, genreGroups[genre], currentMatrixValues)).join("")
      : `<div class="empty-state">No hay corridas con audio agrupadas por género. Renderiza WAV/MP3 primero.</div>`;
  }
  updateGeneratedSelectionState();
  const generatedEvaluations = (state.evaluations || []).filter((item) => item.source === "generated_results");
  renderSelect(
    evaluationSelect,
    generatedEvaluations,
    "No hay reportes calculados",
    (item) => item.evaluation_id,
    (item) => `${item.evaluation_id} · ${item.status || "sin estado"}`,
  );
  if (evaluationSelect && !evaluationSelect.value && generatedEvaluations[0]?.evaluation_id) {
    evaluationSelect.value = generatedEvaluations[0].evaluation_id;
  }
  renderGeneratedEvaluationSummary();
}

function updateGeneratedSelectionState() {
  const selectionStatus = document.querySelector("#generatedMetricsSelectionStatus");
  const runButton = document.querySelector("#generatedMetricsRunButton");
  const selections = readGeneratedEvaluationSelections();
  const selectedTotal = selections.reduce(
    (sum, selection) => sum + selection.rows.reduce((rowSum, row) => rowSum + Number(row.limit || 0), 0),
    0,
  );
  if (selectionStatus) {
    selectionStatus.textContent = selectedTotal
      ? `Seleccionadas ${selectedTotal} canciones en ${selections.length} género(s).`
      : "Escribe cuántas canciones evaluar por género (o usa “Seleccionar por género” para llenarlos todos de una vez).";
  }
  if (runButton) {
    runButton.disabled = state.busy || selectedTotal <= 0;
  }
}

function generatedGenreGroups() {
  const direct = state.evaluationSources?.genre_groups;
  if (direct && Object.keys(direct).length) return direct;
  const genreGroups = {};
  for (const source of state.evaluationSources?.sources || []) {
    for (const [genre, summary] of Object.entries(source.genres || {})) {
      const group = genreGroups[genre] || {
        genre,
        label: genre,
        sources: [],
        total_audio_ready: 0,
        total_missing_audio: 0,
        total_candidates: 0,
      };
      group.sources.push({
        source_type: source.source_type,
        source_id: source.source_id,
        label: source.label,
        created_at: source.created_at,
        generation_mode: source.generation_mode,
        generation_mode_label: source.generation_mode_label,
        source_group: source.source_group,
        audio_ready_count: summary.audio_ready_count || 0,
        missing_audio_count: summary.missing_audio_count || 0,
        total_candidates: summary.total_candidates || 0,
        max_selectable: summary.max_selectable || 0,
      });
      group.total_audio_ready += Number(summary.audio_ready_count || 0);
      group.total_missing_audio += Number(summary.missing_audio_count || 0);
      group.total_candidates += Number(summary.total_candidates || 0);
      genreGroups[genre] = group;
    }
  }
  return genreGroups;
}

function readGeneratedMatrixValues() {
  const values = {};
  document.querySelectorAll(".generated-matrix-input").forEach((input) => {
    const key = `${input.dataset.genre || ""}::${input.dataset.sourceId || ""}`;
    values[key] = Number.parseInt(input.value || "0", 10) || 0;
  });
  return values;
}

function renderGeneratedGenreMatrix(genre, group, selectedValues = {}) {
  const realCount = Number(state.evaluationAvailability?.counts?.[genre] || 0);
  const sources = (group.sources || [])
    .filter((source) => Number(source.audio_ready_count || 0) > 0)
    .sort((a, b) => String(b.created_at || "").localeCompare(String(a.created_at || "")));
  const ready = Number(group.total_audio_ready || 0);
  const missing = Number(group.total_missing_audio || 0);
  const maxSelectable = sources.reduce(
    (sum, source) => sum + Number(source.max_selectable || source.audio_ready_count || 0),
    0,
  );
  const cap = realCount > 0 ? maxSelectable : 0;
  // Total ya elegido (preservado de un render anterior o de edición manual).
  const currentTotal = sources.reduce(
    (sum, source) =>
      sum +
      Math.min(
        Number(source.max_selectable || source.audio_ready_count || 0),
        Math.max(0, Number(selectedValues[`${genre}::${source.source_id}`] || 0)),
      ),
    0,
  );
  const rows = sources.length
    ? sources
        .map((source) => {
          const max = Number(source.max_selectable || source.audio_ready_count || 0);
          const mode = source.generation_mode_label || (source.source_group === "fusion" ? "Fusión explícita" : "Generación normal");
          const selected = realCount > 0 ? Math.min(max, Math.max(0, Number(selectedValues[`${genre}::${source.source_id}`] || 0))) : 0;
          return `
            <tr>
              <td>
                <strong>${escapeHtml(mode)}</strong>
                <small>${escapeHtml(compactSourceLabel(source))}</small>
              </td>
              <td>${Number(source.audio_ready_count || 0)}</td>
              <td>${Number(source.missing_audio_count || 0)}</td>
              <td>
                <input class="generated-matrix-input" data-genre="${escapeHtml(genre)}" data-source-id="${escapeHtml(source.source_id)}" data-source-type="${escapeHtml(source.source_type || "ranking")}" type="number" min="0" max="${max}" value="${selected}" ${realCount > 0 ? "" : "disabled"} title="Cantidad de canciones ${escapeHtml(genre)} a tomar de esta corrida" />
              </td>
            </tr>
          `;
        })
        .join("")
    : `<tr><td colspan="4">No hay WAV/MP3 listos para este género.</td></tr>`;
  return `
    <section class="metrics-genre-block">
      <div class="metrics-genre-heading">
        <div>
          <strong>${escapeHtml(genre)}</strong>
          <small>${ready} con audio · ${realCount} reales disponibles${missing ? ` · ${missing} requieren render` : ""}</small>
        </div>
        <span>${realCount <= 0 ? "Sin originales" : ready > 0 ? "Listo" : "Sin audio"}</span>
      </div>
      <label class="genre-total-control">
        <span>Canciones a evaluar</span>
        <input class="genre-total-input" data-genre="${escapeHtml(genre)}" type="number" min="0" max="${cap}" value="${Math.min(cap, currentTotal)}" ${cap > 0 ? "" : "disabled"} />
        <small>${
          cap > 0
            ? `Se eligen automáticamente las de mejor puntaje, repartidas entre corridas (máx ${cap}). Se emparejan con el mismo número de originales reales.`
            : "No hay audio real de este género para comparar."
        }</small>
      </label>
      <details class="advanced">
        <summary>Elegir manualmente por corrida</summary>
        <div class="table-wrap">
          <table class="metrics-table metrics-matrix-table">
            <thead>
              <tr><th>Corrida</th><th>Con audio</th><th>Sin audio</th><th>A evaluar</th></tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      </details>
    </section>
  `;
}

// Reparte un total por género entre sus corridas (las más recientes primero),
// escribiendo en los inputs ocultos de la matriz (fuente de verdad para el envío).
function distributeGenreTotal(genre, total) {
  const inputs = Array.from(document.querySelectorAll(`.generated-matrix-input[data-genre="${(window.CSS && CSS.escape) ? CSS.escape(genre) : genre}"]`));
  let remaining = Math.max(0, Number(total) || 0);
  for (const input of inputs) {
    if (input.disabled) {
      input.value = 0;
      continue;
    }
    const max = Number.parseInt(input.max || "0", 10) || 0;
    const take = Math.min(max, remaining);
    input.value = take;
    remaining -= take;
  }
}

// Refleja en el input por género la suma de lo elegido manualmente en su matriz.
function syncGenreTotalFromMatrix(genre) {
  const totalInput = document.querySelector(`.genre-total-input[data-genre="${(window.CSS && CSS.escape) ? CSS.escape(genre) : genre}"]`);
  if (!totalInput) return;
  const sum = Array.from(
    document.querySelectorAll(`.generated-matrix-input[data-genre="${(window.CSS && CSS.escape) ? CSS.escape(genre) : genre}"]`),
  ).reduce((acc, input) => acc + (Number.parseInt(input.value || "0", 10) || 0), 0);
  totalInput.value = sum;
}

function generatedSourceGroups() {
  const fallback = {
    normal: { label: "Generación normal", sources: [], total_audio_ready: 0, total_missing_audio: 0 },
    fusion: { label: "Fusión explícita de géneros", sources: [], total_audio_ready: 0, total_missing_audio: 0 },
    renders: { label: "Renders sueltos", sources: [], total_audio_ready: 0, total_missing_audio: 0 },
  };
  const groups = state.evaluationSources?.groups || {};
  for (const key of Object.keys(fallback)) {
    fallback[key] = { ...fallback[key], ...(groups[key] || {}) };
  }
  if (!state.evaluationSources?.groups) {
    for (const source of state.evaluationSources?.sources || []) {
      const key = source.source_group || (source.generation_mode === "genre_fusion" ? "fusion" : source.source_type === "renders" ? "renders" : "normal");
      fallback[key].sources.push(source);
      fallback[key].total_audio_ready += Number(source.audio_ready_count || 0);
      fallback[key].total_missing_audio += Number(source.missing_audio_count || 0);
    }
  }
  return fallback;
}

function compactSourceLabel(source) {
  const ready = Number(source.audio_ready_count || 0);
  const missing = Number(source.missing_audio_count || 0);
  const shortId = String(source.source_id || "").replace(/^\d{8}-\d{6}-/, "");
  return `${source.generation_mode_label || "Generación"} · ${shortId} · ${ready} listas${missing ? `, ${missing} sin audio` : ""}`;
}

function currentEvaluation(preferredSelector = "#generatedEvaluationSelect") {
  const fallbackSelector = preferredSelector === "#generatedEvaluationSelect" ? "#metricsEvaluationSelect" : "#generatedEvaluationSelect";
  const selected = valueOf(preferredSelector) || valueOf(fallbackSelector);
  return (state.evaluations || []).find((item) => item.evaluation_id === selected) || (state.evaluations || [])[0];
}

function renderEvaluationSummary() {
  const summaryEl = document.querySelector("#metricsSummary");
  const filesEl = document.querySelector("#metricsFiles");
  if (!summaryEl || !filesEl) return;
  const evaluation = currentEvaluation("#metricsEvaluationSelect");
  if (!evaluation) {
    summaryEl.innerHTML = "Sin evaluación real calculada todavía.";
    filesEl.innerHTML = "";
    return;
  }
  summaryEl.innerHTML = renderEvaluationReport(evaluation, {
    scope: "batch",
    emptyText: "Cuando calcules métricas aparecerá el reporte completo: general, por género y por canción.",
  });
  const links = [];
  if (evaluation.report_download_url) links.push({ label: "Reporte completo", url: evaluation.report_download_url });
  if (evaluation.manifest_download_url) links.push({ label: "Selección evaluada", url: evaluation.manifest_download_url });
  filesEl.innerHTML = links.length
    ? links.map((link) => `<a href="${API_BASE}${encodeURI(link.url)}" target="_blank" rel="noreferrer">${escapeHtml(link.label)}</a>`).join("")
    : `<small class="muted">Cuando calcules métricas aparecerán descargas del reporte.</small>`;
}

function renderGeneratedEvaluationSummary() {
  const summaryEl = document.querySelector("#generatedMetricsSummary");
  const filesEl = document.querySelector("#generatedMetricsFiles");
  if (!summaryEl || !filesEl) return;
  const selected = valueOf("#generatedEvaluationSelect");
  const evaluation = (state.evaluations || []).find(
    (item) => item.evaluation_id === selected && item.source === "generated_results",
  );
  if (!evaluation) {
    summaryEl.innerHTML = "Sin métricas calculadas todavía.";
    filesEl.innerHTML = "";
    return;
  }
  summaryEl.innerHTML = renderEvaluationReport(evaluation, {
    scope: "generated",
    emptyText: "Sin métricas calculadas todavía.",
  });
  const links = [];
  if (evaluation.report_download_url) links.push({ label: "Reporte completo", url: evaluation.report_download_url });
  if (evaluation.manifest_download_url) links.push({ label: "Selección evaluada", url: evaluation.manifest_download_url });
  filesEl.innerHTML = links.length
    ? links.map((link) => `<a href="${API_BASE}${encodeURI(link.url)}" target="_blank" rel="noreferrer">${escapeHtml(link.label)}</a>`).join("")
    : `<small class="muted">Cuando calcules métricas aparecerán descargas del reporte.</small>`;
}

function renderEvaluationReport(evaluation, options = {}) {
  const summary = evaluation?.summary || {};
  const report = state.evaluationReports?.[evaluation?.evaluation_id] || {};
  const scope = options.scope || String(evaluation?.evaluation_id || "evaluation");
  const hasReport = Object.keys(summary).length || Object.keys(report || {}).length;
  if (!hasReport) return options.emptyText || "Sin métricas calculadas todavía.";
  return `
    ${renderMetricsDashboard(summary)}
    ${renderGenreComparison(report, summary)}
    ${renderGenreSummaryDashboard(report, summary)}
    ${renderMetricsGlossary()}
    ${renderPairMetricsCards(report, scope)}
  `;
}

function genreComparisonData(report, summary) {
  return report?.metrics?.genre_summary?.comparison || summary?.genre_comparison || {};
}

// Tabla coherente que enfrenta audio REAL vs GENERADO por genero. El audio real se
// midio a la misma duracion que el generado, de modo que tempo, probabilidad de
// genero y sonoridad son directamente comparables, mas las distancias FAD/KLD/similitud.
function renderGenreComparison(report, summary) {
  const data = genreComparisonData(report, summary);
  const genres = Object.keys(data).sort();
  if (!genres.length) return "";
  const fmt = (value, suffix = "") =>
    value === null || value === undefined || Number.isNaN(Number(value)) ? "--" : `${formatNumber(value)}${suffix}`;
  const rows = genres
    .map((genre) => {
      const group = data[genre] || {};
      const real = group.real || {};
      const generated = group.generated || {};
      const distances = group.distances || {};
      const seconds = group.comparison_seconds_mean;
      return `
        <tr>
          <th scope="row">
            <strong>${escapeHtml(genre)}</strong>
            <small>${Number(group.pairs || 0)} canciones${seconds === null || seconds === undefined ? "" : ` · ${formatNumber(seconds)} s`}</small>
          </th>
          <td>${fmt(real.tempo_mean)}</td>
          <td class="gen-col">${fmt(generated.tempo_mean)}</td>
          <td>${fmt(real.target_probability_mean)}</td>
          <td class="gen-col">${fmt(generated.target_probability_mean)}</td>
          <td>${fmt(real.rms_mean)}</td>
          <td class="gen-col">${fmt(generated.rms_mean)}</td>
          <td class="dist-col">${fmt(distances.fad_mean)}</td>
          <td class="dist-col">${fmt(distances.kld_mean)}</td>
          <td class="dist-col">${fmt(distances.audio_similarity_mean)}</td>
        </tr>
      `;
    })
    .join("");
  return `
    <section class="genre-comparison-panel">
      <h4>Comparación directa: audio real vs generado por género</h4>
      <p class="muted">
        Cada género enfrenta el audio <b>real</b> contra el <b>generado</b>, medidos a la misma duración
        para que la comparación sea justa. Tempo en BPM y probabilidad de género; FAD/KLD bajos y similitud alta indican mayor cercanía al audio real.
      </p>
      <div class="genre-comparison-scroll">
        <table class="genre-comparison-table">
          <thead>
            <tr>
              <th rowspan="2">Género</th>
              <th colspan="2">Tempo (BPM)</th>
              <th colspan="2">Prob. de género</th>
              <th colspan="2">Sonoridad (RMS)</th>
              <th colspan="3">Distancia real ↔ generado</th>
            </tr>
            <tr>
              <th>Real</th><th class="gen-col">Generado</th>
              <th>Real</th><th class="gen-col">Generado</th>
              <th>Real</th><th class="gen-col">Generado</th>
              <th class="dist-col metric-help" data-help="${escapeHtml(metricHelp("FAD"))}">FAD</th>
              <th class="dist-col metric-help" data-help="${escapeHtml(metricHelp("KLD"))}">KLD</th>
              <th class="dist-col metric-help" data-help="${escapeHtml(metricHelp("Similitud"))}">Similitud</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </section>
  `;
}

function renderGenreSummaryDashboard(report, summary) {
  const genreSummary = report?.metrics?.genre_summary?.genres || summary.genre_summary || {};
  const genres = Object.keys(genreSummary).sort();
  if (!genres.length) {
    return `<p class="muted">Cuando el reporte incluya comparación por género aparecerá aquí.</p>`;
  }
  return `
    <section class="genre-summary-dashboard">
      <h4>Similitud general por género</h4>
      <div class="genre-summary-grid">
        ${genres
          .map((genre) => {
            const row = genreSummary[genre] || {};
            const similarity = row.audio_similarity_mean;
            const quality = row.midi_quality_mean;
            return `
              <article class="genre-summary-card">
                <strong>${escapeHtml(genre)}</strong>
                <span>${Number(row.pairs || 0)} canciones evaluadas</span>
                <div class="mini-metric metric-help" data-help="${escapeHtml(metricHelp("Similitud"))}"><b>${similarity === null || similarity === undefined ? "--" : formatNumber(similarity)}</b><small>similitud audio</small></div>
                <div class="mini-metric metric-help" data-help="${escapeHtml(metricHelp("Calidad MIDI"))}"><b>${quality === null || quality === undefined ? "--" : formatNumber(quality)}</b><small>calidad MIDI</small></div>
                <small class="metric-help" data-help="${escapeHtml(metricHelp("Diferencia tempo"))}">${row.tempo_delta_mean === null || row.tempo_delta_mean === undefined ? "Tempo sin dato" : `Diferencia tempo: ${formatNumber(row.tempo_delta_mean)} BPM`} · ${row.errors || 0} advertencias</small>
              </article>
            `;
          })
          .join("")}
      </div>
    </section>
  `;
}

function metricValue(value, suffix = "") {
  return value === null || value === undefined || Number.isNaN(Number(value)) ? "--" : `${formatNumber(value)}${suffix}`;
}

function absMetric(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.abs(number) : null;
}

function metricHelp(label) {
  const help = {
    "Calidad general": "Combinación local de calidad MIDI, validez MIDI y reward disponible. Mayor es mejor.",
    "FAD": "Distancia aproximada entre la canción original y la generada usando descriptores mel. Menor es mejor.",
    "KLD": "Diferencia entre probabilidades de género de la original y la generada. Menor es mejor.",
    "Validez MIDI": "Indica si el MIDI generado es válido. 1 significa válido.",
    "Similitud": "Parecido acústico entre la canción generada y la original emparejada. Mayor es mejor.",
    "Duración": "Segundos de audio generado y original.",
    "Diferencia duración": "Distancia en segundos entre ambas duraciones. Menor suele ser mejor.",
    "Tempo": "BPM aproximado de la generada y de la original.",
    "Diferencia tempo": "Distancia en BPM entre ambas canciones. Menor suele indicar mayor coherencia.",
    "Calidad MIDI": "Puntaje de validez y estructura musical del MIDI generado. Mayor es mejor.",
    "Densidad": "Notas por segundo. Ayuda a detectar piezas vacías o saturadas.",
    "Diversidad pitch": "Variedad de clases de nota usadas. Mayor suele indicar más riqueza melódica/armónica.",
    "Diversidad rítmica": "Variedad temporal de eventos. Mayor suele indicar menos monotonía.",
    "Prob. género": "Confianza del clasificador de que el audio corresponde al género esperado.",
    "Reward": "Puntaje interno del ranking. Sirve para ordenar candidatas; no sustituye FAD/KLD.",
  };
  return help[label] || "";
}

function renderPairMetric(label, value, suffix = "") {
  return `
    <div class="pair-metric metric-help" title="${escapeHtml(metricHelp(label))}" data-help="${escapeHtml(metricHelp(label))}">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(metricValue(value, suffix))}</strong>
    </div>
  `;
}

function renderPairMetricGroup(title, metricsHtml) {
  return `
    <div class="pair-metric-section">
      <span>${escapeHtml(title)}</span>
      <div class="pair-metric-grid">${metricsHtml}</div>
    </div>
  `;
}

function renderMetricsGlossary() {
  const rows = [
    ["FAD", "Distancia global contra música real. Menor es mejor."],
    ["KLD", "Diferencia entre distribuciones de género real y generado. Menor es mejor."],
    ["Similitud de audio", "Parecido acústico entre cada generada y su original emparejada. Mayor es mejor."],
    ["Diferencia de tempo", "Distancia en BPM entre ambas canciones. Menor suele ser mejor."],
    ["Diferencia de duración", "Distancia en segundos entre ambas canciones. Menor indica mayor consistencia."],
    ["Calidad MIDI", "Validez y estructura del MIDI generado. Mayor es mejor."],
    ["Densidad de notas", "Cantidad de notas por segundo. Detecta piezas vacías o saturadas."],
    ["Diversidad de pitch", "Variedad de notas usadas. Mayor suele indicar más riqueza."],
    ["Diversidad rítmica", "Variedad temporal de eventos. Mayor suele indicar menos monotonía."],
    ["Probabilidad de género", "Confianza del clasificador de que el audio pertenece al género esperado."],
    ["Reward", "Puntaje interno de ranking. No sustituye FAD/KLD/CLAP."],
  ];
  return `
    <details class="metrics-explainer">
      <summary>Qué significa cada métrica</summary>
      <div class="metrics-explainer-grid">
        ${rows.map(([name, text]) => `<div><strong>${escapeHtml(name)}</strong><p>${escapeHtml(text)}</p></div>`).join("")}
      </div>
    </details>
  `;
}

function renderPairMetricsControls(rows, scope) {
  const genres = [...new Set(rows.map((row) => row.genre || "unknown"))].sort();
  const selectedGenre = pairFilterValue(scope, "genre");
  const selectedStatus = pairFilterValue(scope, "status");
  const selectedSort = pairFilterValue(scope, "sort") || "similarity";
  return `
    <div class="pair-controls">
      <label class="field compact-field">
        <span>Filtrar género</span>
        <select class="pair-metrics-control" data-pair-filter="genre" data-pair-scope="${escapeHtml(scope)}">
          <option value="">Todos</option>
          ${genres.map((genre) => `<option value="${escapeHtml(genre)}" ${selectedGenre === genre ? "selected" : ""}>${escapeHtml(genre)}</option>`).join("")}
        </select>
      </label>
      <label class="field compact-field">
        <span>Estado</span>
        <select class="pair-metrics-control" data-pair-filter="status" data-pair-scope="${escapeHtml(scope)}">
          <option value="">Todos</option>
          <option value="ok" ${selectedStatus === "ok" ? "selected" : ""}>OK</option>
          <option value="warning" ${selectedStatus === "warning" ? "selected" : ""}>Con advertencias</option>
        </select>
      </label>
      <label class="field compact-field">
        <span>Ordenar por</span>
        <select class="pair-metrics-control" data-pair-filter="sort" data-pair-scope="${escapeHtml(scope)}">
          <option value="similarity" ${selectedSort === "similarity" ? "selected" : ""}>Similitud</option>
          <option value="quality" ${selectedSort === "quality" ? "selected" : ""}>Calidad MIDI</option>
          <option value="tempo" ${selectedSort === "tempo" ? "selected" : ""}>Diferencia tempo</option>
        </select>
      </label>
    </div>
  `;
}

function pairFilterValue(scope, filter) {
  const control = Array.from(document.querySelectorAll(".pair-metrics-control")).find(
    (item) => item.dataset.pairScope === scope && item.dataset.pairFilter === filter,
  );
  return control?.value || "";
}

function filteredPairRows(rows, scope) {
  const genre = pairFilterValue(scope, "genre");
  const status = pairFilterValue(scope, "status");
  const sort = pairFilterValue(scope, "sort") || "similarity";
  let filtered = [...rows];
  if (genre) filtered = filtered.filter((row) => row.genre === genre);
  if (status === "ok") filtered = filtered.filter((row) => !(row.errors || []).length);
  if (status === "warning") filtered = filtered.filter((row) => (row.errors || []).length);
  const metricNumber = (row, path, fallback = -Infinity) => {
    const value = path.reduce((current, key) => (current && current[key] !== undefined ? current[key] : undefined), row);
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  };
  filtered.sort((a, b) => {
    if (sort === "quality") return metricNumber(b, ["metrics", "midi_quality_score"]) - metricNumber(a, ["metrics", "midi_quality_score"]);
    if (sort === "tempo") return Math.abs(metricNumber(a, ["metrics", "tempo_delta"], Infinity)) - Math.abs(metricNumber(b, ["metrics", "tempo_delta"], Infinity));
    return metricNumber(b, ["metrics", "audio_similarity"]) - metricNumber(a, ["metrics", "audio_similarity"]);
  });
  return filtered;
}

function renderPairMetricsCards(report, scope = "evaluation") {
  const rows = report?.metrics?.pairs?.rows || [];
  if (!rows.length) {
    return `<p class="muted">El detalle canción contra original aparecerá cuando calcules métricas con la nueva matriz.</p>`;
  }
  const visibleRows = filteredPairRows(rows, scope);
  return `
    <section class="pair-metrics-panel">
      <h4>Detalle por canción generada vs original</h4>
      <p class="muted">Pasa el mouse por cualquier métrica para ver qué significa.</p>
      ${renderPairMetricsControls(rows, scope)}
      <div class="pair-card-list">
        ${visibleRows.length
          ? visibleRows
              .map((row) => {
                const metrics = row.metrics || {};
                const source = row.source || {};
                const generated = row.generated || {};
                const original = row.original || {};
                const errors = row.errors || [];
                const generatedUrl = generated.audio_url ? `${API_BASE}${encodeURI(generated.audio_url)}` : "";
                const originalUrl = original.audio_url ? `${API_BASE}${encodeURI(original.audio_url)}` : "";
                const midiQuality = metrics.midi_quality_score ?? metrics.midi?.quality_score;
                return `
                  <article class="pair-card">
                    <header class="pair-card-header">
                      <div>
                        <span>${escapeHtml(row.genre || "unknown")}</span>
                        <strong>${escapeHtml(row.candidate_id || row.track_id || "candidata")}</strong>
                        <small>${escapeHtml(source.generation_mode_label || "Generación")} · rank ${row.rank ?? "--"} · score ${metricValue(row.score)}</small>
                      </div>
                      ${errors.length ? `<b class="status-warning">Requiere atención</b>` : `<b class="status-ok">OK</b>`}
                    </header>
                    <div class="pair-audio-grid">
                      <div class="audio-compare-box">
                        <span>Generada</span>
                        ${generatedUrl ? `<audio controls preload="none" src="${generatedUrl}"></audio>` : "<small>Sin audio generado</small>"}
                      </div>
                      <div class="audio-compare-box">
                        <span>Original</span>
                        ${originalUrl ? `<audio controls preload="none" src="${originalUrl}"></audio>` : "<small>Sin audio original</small>"}
                      </div>
                    </div>
                    ${renderPairMetricGroup(
                      "Métricas principales por canción",
                      [
                        renderPairMetric("Calidad general", metrics.quality_general),
                        renderPairMetric("FAD", metrics.fad),
                        renderPairMetric("KLD", metrics.kld),
                        renderPairMetric("Validez MIDI", metrics.valid_midi_score ?? (metrics.valid_midi === true ? 1 : metrics.valid_midi === false ? 0 : null)),
                        renderPairMetric("Duración", metrics.duration_seconds ?? metrics.generated_duration_seconds, "s"),
                        renderPairMetric("Tempo", metrics.tempo ?? metrics.generated_tempo, " BPM"),
                        renderPairMetric("Diversidad pitch", metrics.pitch_diversity ?? metrics.pitch_class_diversity),
                        renderPairMetric("Diversidad rítmica", metrics.rhythm_diversity),
                      ].join(""),
                    )}
                    ${renderPairMetricGroup(
                      "Métricas complementarias",
                      [
                        renderPairMetric("Similitud", metrics.audio_similarity),
                        renderPairMetric("Diferencia duración", absMetric(metrics.duration_delta_seconds), "s"),
                        renderPairMetric("Diferencia tempo", absMetric(metrics.tempo_delta), " BPM"),
                        renderPairMetric("Densidad", metrics.note_density_per_second),
                        renderPairMetric("Prob. género", metrics.generated_target_probability),
                        renderPairMetric("Reward", metrics.reward_score),
                      ].join(""),
                    )}
                    ${
                      errors.length
                        ? `<ul class="pair-errors">${errors.map((error) => `<li>${escapeHtml(error.metric || "métrica")}: ${escapeHtml(error.error || "sin detalle")}</li>`).join("")}</ul>`
                        : ""
                    }
                  </article>
                `;
              })
              .join("")
          : `<p class="muted">No hay canciones con los filtros seleccionados.</p>`}
      </div>
    </section>
  `;
}

function metricStatus(value, higherIsBetter = true) {
  const number = Number.parseFloat(value);
  if (!Number.isFinite(number)) return "Sin dato";
  if (higherIsBetter === null) return "Informativo";
  if (higherIsBetter) {
    if (number >= 0.75) return "Bueno";
    if (number >= 0.45) return "Aceptable";
    return "Requiere mejora";
  }
  if (number <= 0.5) return "Bueno";
  if (number <= 2.0) return "Aceptable";
  return "Requiere mejora";
}

function metricPercent(value, higherIsBetter = true) {
  const number = Number.parseFloat(value);
  if (!Number.isFinite(number)) return 0;
  if (higherIsBetter === null) return 65;
  if (higherIsBetter) return Math.max(0, Math.min(100, number * 100));
  return Math.max(0, Math.min(100, 100 / (1 + Math.max(0, number))));
}

function renderMetricCard(label, value, help, higherIsBetter = true, suffix = "") {
  const shown = value === null || value === undefined ? "--" : `${formatNumber(value)}${suffix}`;
  const pct = metricPercent(value, higherIsBetter);
  return `
    <div class="metric-card-visual metric-help" data-help="${escapeHtml(help)}">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(shown)}</strong>
      <div class="metric-bar"><i style="width:${pct}%"></i></div>
      <small>${escapeHtml(metricStatus(value, higherIsBetter))} · ${escapeHtml(help)}</small>
    </div>
  `;
}

function renderMetricsDashboard(summary) {
  const quality = summary.quality_mean ?? summary.reward_mean ?? summary.valid_midi_rate;
  return `
    <div class="metrics-dashboard-grid">
      ${renderMetricCard("Calidad general", quality, "promedio de calidad/reward", true)}
      ${renderMetricCard("FAD", summary.fad, "menor es más cercano al audio real", false)}
      ${renderMetricCard("KLD", summary.kld, "menor significa distribución más parecida", false)}
      ${renderMetricCard("Validez MIDI", summary.valid_midi_rate, "porcentaje de MIDIs válidos", true)}
      ${renderMetricCard("Duración promedio", summary.duration_mean, "segundos generados", null, "s")}
      ${renderMetricCard("Tempo promedio", summary.tempo_mean, "BPM aproximado", null)}
      ${renderMetricCard("Diversidad pitch", summary.pitch_diversity_mean, "variedad de notas", true)}
      ${renderMetricCard("Diversidad rítmica", summary.rhythm_diversity_mean, "variedad temporal", true)}
    </div>
    ${
      summary.errors?.length
        ? `<p class="metric-warning">Se completó con ${summary.errors.length} advertencia(s). Revisa el reporte completo.</p>`
        : `<p class="muted">FAD/KLD bajos son mejores. Validez, calidad y diversidad altas suelen ser mejores.</p>`
    }
  `;
}

function setMetricsDistribution(mode) {
  const counts = state.evaluationAvailability?.counts || {};
  const recommended = state.evaluationAvailability?.recommended_distribution || {};
  const inputs = document.querySelectorAll(".metrics-distribution-input");
  inputs.forEach((input) => {
    const genre = input.dataset.genre;
    if (mode === "max") input.value = counts[genre] || 0;
    if (mode === "recommended") input.value = recommended[genre] || 0;
    if (mode === "clear") input.value = 0;
  });
}

function readMetricsDistribution() {
  const distribution = {};
  document.querySelectorAll(".metrics-distribution-input").forEach((input) => {
    const genre = input.dataset.genre;
    const value = Number.parseInt(input.value || "0", 10);
    if (genre && Number.isFinite(value) && value > 0) distribution[genre] = value;
  });
  return distribution;
}

async function generateEvaluationBatch() {
  const modelPath = valueOf("#metricsModelSelect");
  if (!modelPath) throw new Error("Selecciona un modelo generativo para crear el lote.");
  const distribution = readMetricsDistribution();
  if (!Object.keys(distribution).length) throw new Error("Indica al menos una cantidad a generar por género.");
  const result = await postJob("/api/jobs/evaluation/generate-batch", {
    model_path: modelPath,
    distribution,
    duration_seconds: optionalFloat("#metricsDuration") || 30,
    output_name: "evaluation_batch",
    render_audio: checked("#metricsRenderAudio"),
    render_engine: valueOf("#metricsRenderEngine") || "auto",
    export_mp3: checked("#metricsExportMp3"),
    target_total: Object.values(distribution).reduce((sum, value) => sum + Number(value || 0), 0),
  });
  renderJson(els.outputJson, result);
}

async function runEvaluationMetrics() {
  const evaluationId = valueOf("#metricsEvaluationSelect") || currentEvaluation("#metricsEvaluationSelect")?.evaluation_id;
  if (!evaluationId) throw new Error("Selecciona un lote de evaluación primero.");
  const metrics = ["fad", "kld", "tempo", "midi"];
  if (checked("#metricsUseClap")) metrics.push("clap");
  const result = await postJob("/api/jobs/evaluation/run", {
    evaluation_id: evaluationId,
    metrics,
    fad_extractor: "mel",
    train_classifier_if_missing: true,
  });
  renderJson(els.outputJson, result);
}

// ── Proyección de embeddings (t-SNE / PCA) ────────────────────────────────────
const projectionState = { data: null, method: "tsne", runPath: "" };

async function runEmbeddingProjection() {
  const runPath = valueOf("#embeddingProjectionRun");
  const statusEl = document.querySelector("#embeddingProjectionStatus");
  const plot = document.querySelector("#embeddingProjectionPlot");
  if (!runPath) {
    if (statusEl) statusEl.textContent = 'Primero crea embeddings por género (paso "Crear embeddings por género").';
    return;
  }
  const maxPerGenre = Math.max(3, Number.parseInt(valueOf("#embeddingProjectionMax") || "60", 10) || 60);
  if (statusEl) statusEl.textContent = "Calculando proyección (codificando embeddings con el Token-VAE)...";
  if (plot) plot.innerHTML = "";
  try {
    const data = await getJson(
      `/api/metrics/embedding-projection?run_path=${encodeURIComponent(runPath)}&max_per_genre=${maxPerGenre}`,
    );
    projectionState.data = data;
    projectionState.runPath = runPath;
    if (projectionState.method === "tsne" && !data.has_tsne) projectionState.method = "pca";
    renderEmbeddingProjection();
  } catch (error) {
    if (statusEl) statusEl.textContent = `No se pudo proyectar: ${error.message || error}`;
  }
}

function setProjectionMethod(method) {
  projectionState.method = method;
  document.querySelectorAll("#embeddingProjectionMethodToggle .method-tab").forEach((tab) => {
    tab.classList.toggle("is-active", tab.dataset.projectionMethod === method);
  });
  if (projectionState.data) renderEmbeddingProjection();
}

function renderEmbeddingProjection() {
  const data = projectionState.data;
  const statusEl = document.querySelector("#embeddingProjectionStatus");
  const plot = document.querySelector("#embeddingProjectionPlot");
  const legend = document.querySelector("#embeddingProjectionLegend");
  if (!data || !plot) return;
  let method = projectionState.method || "tsne";
  let points = method === "pca" ? data.pca : data.tsne;
  if (method === "tsne" && (!points || !points.length)) {
    method = "pca";
    points = data.pca;
  }
  if (!points || !points.length) {
    plot.innerHTML = `<p class="muted">No hay puntos suficientes para proyectar.</p>`;
    if (legend) legend.innerHTML = "";
    return;
  }
  plot.innerHTML = renderScatterSvg(points, data.genre_colors || {});
  if (statusEl) {
    const variance = (data.pca_explained_variance || []).map((value) => `${(value * 100).toFixed(1)}%`).join(" / ");
    const methodLabel = method === "pca" ? `PCA (varianza por eje: ${variance || "--"})` : "t-SNE (estructura local)";
    statusEl.textContent = `${methodLabel} · ${data.n_tracks} pistas + ${data.n_centroids} centroides · ${(data.genres || []).length} géneros · dim. latente ${data.latent_dim}`;
  }
  if (legend) {
    const genreItems = (data.genres || [])
      .map(
        (genre) =>
          `<span class="legend-item"><i style="background:${(data.genre_colors || {})[genre] || "#888"}"></i>${escapeHtml(genre)}</span>`,
      )
      .join("");
    legend.innerHTML = `${genreItems}<span class="legend-item legend-shape">◆ centroide del género · ● pista</span>`;
  }
}

function renderScatterSvg(points, colors) {
  const width = 640;
  const height = 440;
  const pad = 30;
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const sx = (value) => pad + ((value - minX) / spanX) * (width - 2 * pad);
  const sy = (value) => height - pad - ((value - minY) / spanY) * (height - 2 * pad);
  const dots = points
    .map((point) => {
      const color = colors[point.genre] || "#888";
      const cx = sx(point.x).toFixed(1);
      const cy = sy(point.y).toFixed(1);
      const title = `${escapeHtml(point.genre)} · ${escapeHtml(point.label || "")}`;
      if (point.kind === "centroid") {
        const s = 8;
        return `<path d="M ${cx} ${Number(cy) - s} L ${Number(cx) + s} ${cy} L ${cx} ${Number(cy) + s} L ${Number(cx) - s} ${cy} Z" fill="${color}" stroke="#111" stroke-width="1.3" opacity="0.95"><title>${title}</title></path>`;
      }
      return `<circle cx="${cx}" cy="${cy}" r="4" fill="${color}" fill-opacity="0.72" stroke="#fff" stroke-width="0.6"><title>${title}</title></circle>`;
    })
    .join("");
  return `<svg viewBox="0 0 ${width} ${height}" class="scatter-svg" role="img" aria-label="Proyección 2D de embeddings por género">
    <rect x="0.5" y="0.5" width="${width - 1}" height="${height - 1}" fill="#ffffff" stroke="#e3e0d7" />
    ${dots}
  </svg>`;
}

// ── Mapas t-SNE de tokens EnCodec / T5 (reutiliza genre_tsne.py en el backend) ─
const tokenTsneState = { data: null, modality: null };

async function runTokenTsne() {
  const statusEl = document.querySelector("#tokenTsneStatus");
  const plot = document.querySelector("#tokenTsnePlot");
  const maxPoints = Math.max(3, Number.parseInt(valueOf("#tokenTsneMax") || "300", 10) || 300);
  const knn = Math.max(1, Number.parseInt(valueOf("#tokenTsneKnn") || "5", 10) || 5);
  const drawEdges = checked("#tokenTsneEdges");
  if (statusEl) statusEl.textContent = "Calculando t-SNE de tokens (cargando encodings EnCodec/T5)...";
  if (plot) plot.innerHTML = "";
  try {
    const data = await getJson(
      `/api/metrics/token-tsne?max_points=${maxPoints}&knn=${knn}&draw_edges=${drawEdges}`,
    );
    tokenTsneState.data = data;
    if (!tokenTsneState.modality || !(data.modalities || []).includes(tokenTsneState.modality)) {
      tokenTsneState.modality = (data.modalities || [])[0] || null;
    }
    renderTokenTsneToggle();
    renderTokenTsne();
  } catch (error) {
    if (statusEl) statusEl.textContent = `No se pudo generar el mapa: ${error.message || error}`;
  }
}

function renderTokenTsneToggle() {
  const toggle = document.querySelector("#tokenTsneModalityToggle");
  const data = tokenTsneState.data;
  if (!toggle || !data) return;
  const labels = data.modality_labels || {};
  toggle.innerHTML = (data.modalities || [])
    .map(
      (modality) =>
        `<button type="button" class="method-tab${modality === tokenTsneState.modality ? " is-active" : ""}" data-token-modality="${escapeHtml(modality)}">${escapeHtml(labels[modality] || modality)}</button>`,
    )
    .join("");
  toggle.querySelectorAll(".method-tab").forEach((tab) =>
    on(tab, "click", () => {
      tokenTsneState.modality = tab.dataset.tokenModality;
      renderTokenTsneToggle();
      renderTokenTsne();
    }),
  );
}

function renderTokenTsne() {
  const data = tokenTsneState.data;
  const statusEl = document.querySelector("#tokenTsneStatus");
  const plot = document.querySelector("#tokenTsnePlot");
  const legend = document.querySelector("#tokenTsneLegend");
  if (!data || !plot) return;
  const modality = tokenTsneState.modality;
  const map = (data.maps || {})[modality];
  if (!map || !map.points || !map.points.length) {
    plot.innerHTML = `<p class="muted">${escapeHtml(map?.error || "No hay tokens para esta modalidad.")}</p>`;
    if (legend) legend.innerHTML = "";
    return;
  }
  plot.innerHTML = renderTokenMapSvg(map.points, map.edges || [], data.genre_colors || {});
  if (statusEl) {
    const modalityLabel = (data.modality_labels || {})[modality] || modality;
    statusEl.textContent = `${modalityLabel} · ${map.n} tokens · ${(map.edges || []).length} aristas k-NN(${data.knn}) · fuente ${data.source || "encodings"}`;
  }
  if (legend) {
    const genreItems = (data.genres || [])
      .map(
        (genre) =>
          `<span class="legend-item"><i style="background:${(data.genre_colors || {})[genre] || "#888"}"></i>${escapeHtml(genre)}</span>`,
      )
      .join("");
    legend.innerHTML = `${genreItems}<span class="legend-item legend-shape">● palabra (T5) · ▲ sonido (EnCodec)</span>`;
  }
}

function renderTokenMapSvg(points, edges, colors) {
  const width = 640;
  const height = 440;
  const pad = 30;
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const sx = (value) => pad + ((value - minX) / spanX) * (width - 2 * pad);
  const sy = (value) => height - pad - ((value - minY) / spanY) * (height - 2 * pad);
  const edgeLines = (edges || [])
    .map(([a, b]) => {
      const pa = points[a];
      const pb = points[b];
      if (!pa || !pb) return "";
      return `<line x1="${sx(pa.x).toFixed(1)}" y1="${sy(pa.y).toFixed(1)}" x2="${sx(pb.x).toFixed(1)}" y2="${sy(pb.y).toFixed(1)}" stroke="#9ca3af" stroke-width="0.4" stroke-opacity="0.4" />`;
    })
    .join("");
  const nodes = points
    .map((point) => {
      const color = colors[point.genre] || "#888";
      const cx = sx(point.x).toFixed(1);
      const cy = sy(point.y).toFixed(1);
      const title = `${escapeHtml(point.genre)} · ${escapeHtml(point.label || "")}`;
      if (point.modality === "sound") {
        const s = 5;
        return `<path d="M ${cx} ${Number(cy) - s} L ${Number(cx) + s} ${Number(cy) + s} L ${Number(cx) - s} ${Number(cy) + s} Z" fill="${color}" fill-opacity="0.82" stroke="#fff" stroke-width="0.5"><title>${title}</title></path>`;
      }
      return `<circle cx="${cx}" cy="${cy}" r="3.5" fill="${color}" fill-opacity="0.75" stroke="#fff" stroke-width="0.5"><title>${title}</title></circle>`;
    })
    .join("");
  return `<svg viewBox="0 0 ${width} ${height}" class="scatter-svg" role="img" aria-label="Mapa t-SNE de tokens EnCodec/T5">
    <rect x="0.5" y="0.5" width="${width - 1}" height="${height - 1}" fill="#ffffff" stroke="#e3e0d7" />
    ${edgeLines}${nodes}
  </svg>`;
}

// ── Comparativa de modelos (pestaña Gráficas) — carga perezosa ─────────────────
const BENCHMARK_PALETTE = ["#2563EB", "#DC2626", "#059669", "#CA8A04", "#7C3AED"];

async function loadBenchmark(force = false) {
  const content = document.querySelector("#benchmarkContent");
  if (!content) return;
  if (state.benchmark.loading) return;
  if (state.benchmark.loaded && !force) return;
  state.benchmark.loading = true;
  content.innerHTML = `<p class="muted">Cargando comparativa de modelos…</p>`;
  try {
    const base = cloudBase();
    const payload = base
      ? await fetch(`${base}/api/metrics/benchmark`).then((r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.json();
        })
      : await getJson("/api/metrics/benchmark");
    state.benchmark.loaded = true;
    renderBenchmark(payload, content);
  } catch (error) {
    content.innerHTML = `<p class="muted">No se pudo cargar la comparativa: ${escapeHtml(error.message || String(error))}.<br />
      Genera los CSV con <code>python scripts/make_synthetic_results.py</code>, coloca los reales en <code>results/</code>,
      o pulsa <b>Calcular en la nube (GPU)</b> si tienes el backend de Colab conectado.</p>`;
  } finally {
    state.benchmark.loading = false;
  }
}

// Dispara el cálculo real de métricas (FAD/CLAP/KLD/KAD) en la GPU de la nube
// (mismo backend de Colab que la generación) y refresca el benchmark al terminar.
async function runCloudMetrics() {
  const base = cloudBase();
  const content = document.querySelector("#benchmarkContent");
  if (!base) {
    if (content) {
      content.innerHTML = `<p class="muted">Primero conecta el <b>Backend en la nube (GPU)</b> en el Paso 4.</p>`;
    }
    return;
  }
  const created = await fetch(`${base}/api/jobs/eval-metrics`, { method: "POST" }).then((r) => r.json());
  await new Promise((resolve, reject) => {
    const timer = setInterval(async () => {
      try {
        const job = await fetch(`${base}/api/jobs/${created.job_id}`).then((r) => r.json());
        if (content) {
          content.innerHTML = `<p class="muted">${escapeHtml(job.stage || "")}: ${escapeHtml(job.message || "")}</p>`;
        }
        if (job.status === "completed") {
          clearInterval(timer);
          resolve();
        } else if (["failed", "cancelled"].includes(job.status)) {
          clearInterval(timer);
          reject(new Error(job.message || "Falló el cálculo de métricas."));
        }
      } catch (e) {
        clearInterval(timer);
        reject(e);
      }
    }, 2000);
  });
  state.benchmark.loaded = false;
  await loadBenchmark(true);
}

function renderBenchmark(payload, content) {
  const table = payload.table || {};
  const radar = payload.radar || {};
  const stats = payload.stats || {};
  content.innerHTML = `
    <div class="benchmark-grid">
      <section class="benchmark-block">
        <h4>1 · Tabla comparativa (FAD, CLAP, KLD, KAD)</h4>
        ${renderBenchmarkTable(table.rows || [])}
      </section>
      <section class="benchmark-block">
        <h4>3 · Radar — métricas normalizadas (1 = mejor)</h4>
        ${renderBenchmarkRadarSvg(radar)}
        <div class="embedding-legend">${(radar.models || [])
          .map((model, index) => `<span class="legend-item"><i style="background:${BENCHMARK_PALETTE[index % BENCHMARK_PALETTE.length]}"></i>${escapeHtml(model)}</span>`)
          .join("")}</div>
      </section>
    </div>
    <section class="benchmark-block">
      <h4>2 · Análisis estadístico del CLAP-score (Kruskal-Wallis + Dunn)</h4>
      ${renderBenchmarkStats(stats)}
    </section>
  `;
}

function renderBenchmarkTable(rows) {
  if (!rows.length) return `<p class="muted">Sin datos de tabla.</p>`;
  const best = {
    fad_vggish: Math.min(...rows.map((r) => r.fad_vggish)),
    clap_mean: Math.max(...rows.map((r) => r.clap_mean)),
    kld: Math.min(...rows.map((r) => r.kld)),
    kad: Math.min(...rows.map((r) => r.kad)),
  };
  const cell = (value, key, suffix = "") =>
    `<td class="${value === best[key] ? "benchmark-best" : ""}">${formatNumber(value)}${suffix}</td>`;
  const body = rows
    .map(
      (row) => `
        <tr>
          <th scope="row">${escapeHtml(row.model)}</th>
          ${cell(row.fad_vggish, "fad_vggish")}
          <td class="${row.clap_mean === best.clap_mean ? "benchmark-best" : ""}">${formatNumber(row.clap_mean)} ± ${formatNumber(row.clap_std)}</td>
          ${cell(row.kld, "kld")}
          ${cell(row.kad, "kad")}
          <td>${formatNumber(row.overall_rank)}</td>
        </tr>`,
    )
    .join("");
  return `
    <div class="genre-comparison-scroll">
      <table class="genre-comparison-table benchmark-table">
        <thead>
          <tr><th>Modelo</th><th>FAD ↓</th><th>CLAP ↑</th><th>KLD ↓</th><th>KAD ↓</th><th>Rango ↓</th></tr>
        </thead>
        <tbody>${body}</tbody>
      </table>
    </div>
    <p class="muted">Resaltado = mejor valor por métrica. Rango promedio combina las 4 métricas (1 = mejor).</p>
  `;
}

function renderBenchmarkRadarSvg(radar) {
  const metrics = radar.metrics || [];
  const models = radar.models || [];
  const norm = radar.normalized || {};
  const labels = radar.metric_labels || {};
  const dirs = radar.directions || {};
  if (!metrics.length || !models.length) return `<p class="muted">Sin datos para el radar.</p>`;
  const W = 460;
  const H = 430;
  const cx = W / 2;
  const cy = H / 2 + 4;
  const R = 145;
  const n = metrics.length;
  const angle = (i) => -Math.PI / 2 + (i * 2 * Math.PI) / n;
  const pt = (i, r) => [cx + Math.cos(angle(i)) * r, cy + Math.sin(angle(i)) * r];
  let grid = "";
  [0.25, 0.5, 0.75, 1].forEach((level) => {
    const pts = metrics.map((_, i) => pt(i, R * level).map((v) => v.toFixed(1)).join(",")).join(" ");
    grid += `<polygon points="${pts}" fill="none" stroke="#e8e4dc" stroke-width="1" />`;
  });
  let axes = "";
  metrics.forEach((metric, i) => {
    const [x, y] = pt(i, R);
    axes += `<line x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="#d9d7d0" stroke-width="1" />`;
    const [lx, ly] = pt(i, R + 24);
    const dir = dirs[metric] === "min" ? "↓" : "↑";
    axes += `<text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" text-anchor="middle" dominant-baseline="middle" font-size="11" fill="#5a554c">${escapeHtml(labels[metric] || metric)} ${dir}</text>`;
  });
  const polys = models
    .map((model, mi) => {
      const color = BENCHMARK_PALETTE[mi % BENCHMARK_PALETTE.length];
      const pts = metrics
        .map((metric, i) => {
          const value = (norm[metric] && norm[metric][model]) || 0;
          return pt(i, R * value).map((v) => v.toFixed(1)).join(",");
        })
        .join(" ");
      return `<polygon points="${pts}" fill="${color}" fill-opacity="0.10" stroke="${color}" stroke-width="2"><title>${escapeHtml(model)}</title></polygon>`;
    })
    .join("");
  return `<svg viewBox="0 0 ${W} ${H}" class="scatter-svg radar-svg" role="img" aria-label="Radar de comparación de modelos">${grid}${axes}${polys}</svg>`;
}

function renderBenchmarkStats(stats) {
  const overall = stats.overall || {};
  const byGenre = stats.by_genre || {};
  let overallHtml;
  if (overall.h_statistic === null || overall.h_statistic === undefined) {
    overallHtml = `<p class="muted">${escapeHtml(overall.note || "Sin análisis global.")}</p>`;
  } else {
    const dunn = (overall.dunn || [])
      .map(
        (comp) =>
          `<li class="${comp.significant ? "dunn-sig" : ""}">${escapeHtml(comp.model_a)} vs ${escapeHtml(comp.model_b)}: z=${formatNumber(comp.z)}, p<sub>Bonf</sub>=${comp.p_bonferroni < 0.001 ? comp.p_bonferroni.toExponential(2) : formatNumber(comp.p_bonferroni)}${comp.significant ? " ✓" : ""}</li>`,
      )
      .join("");
    overallHtml = `
      <p><b>Global:</b> H=${formatNumber(overall.h_statistic)}, p=${overall.p_value < 0.001 ? overall.p_value.toExponential(2) : formatNumber(overall.p_value)} —
        ${overall.significant ? "<b>diferencias significativas</b>" : "sin diferencia significativa"}.</p>
      ${dunn ? `<p class="muted">Dunn post-hoc (Bonferroni):</p><ul class="dunn-list">${dunn}</ul>` : ""}
    `;
  }
  const genreRows = Object.entries(byGenre)
    .map(([genre, result]) => {
      if (result.p_value === null || result.p_value === undefined) {
        return `<tr><th scope="row">${escapeHtml(genre)}</th><td>--</td><td>${escapeHtml(result.note || "")}</td></tr>`;
      }
      const p = result.p_value < 0.001 ? result.p_value.toExponential(2) : formatNumber(result.p_value);
      return `<tr><th scope="row">${escapeHtml(genre)}</th><td>${formatNumber(result.h_statistic)}</td><td class="${result.significant ? "dunn-sig" : ""}">${p}${result.significant ? " ✓" : ""}</td></tr>`;
    })
    .join("");
  return `
    ${overallHtml}
    <div class="genre-comparison-scroll">
      <table class="genre-comparison-table benchmark-table">
        <thead><tr><th>Género</th><th>H (Kruskal-Wallis)</th><th>p</th></tr></thead>
        <tbody>${genreRows}</tbody>
      </table>
    </div>
    <p class="muted">✓ = significativo a α=0.05. Por género hay 10 clips por modelo, por lo que el post-hoc tiene menos potencia.</p>
  `;
}

function readGeneratedEvaluationSelections() {
  const byGenre = {};
  document.querySelectorAll(".generated-matrix-input").forEach((input) => {
    const genre = input.dataset.genre;
    const sourceId = input.dataset.sourceId;
    const sourceType = input.dataset.sourceType || "ranking";
    const limit = Number.parseInt(input.value || "0", 10);
    const max = Number.parseInt(input.max || "0", 10);
    if (!genre || !sourceId || !Number.isFinite(limit) || limit <= 0) return;
    const safeLimit = Math.min(limit, Math.max(0, max));
    if (safeLimit <= 0) return;
    byGenre[genre] = byGenre[genre] || [];
    byGenre[genre].push({ source_type: sourceType, source_id: sourceId, limit: safeLimit });
  });
  return Object.entries(byGenre).map(([genre, rows]) => ({ genre, rows }));
}

function setGeneratedMatrixSelection(mode) {
  const target = Math.max(1, Number.parseInt(valueOf("#generatedMetricsTarget") || "20", 10) || 20);
  const genreGroups = generatedGenreGroups();
  document.querySelectorAll(".generated-matrix-input").forEach((input) => {
    input.value = 0;
  });
  if (mode === "clear") {
    renderGeneratedEvaluationPanel();
    return;
  }
  for (const [genre, group] of Object.entries(genreGroups)) {
    if (Number(state.evaluationAvailability?.counts?.[genre] || 0) <= 0) continue;
    let remaining = target;
    const sources = (group.sources || [])
      .filter((source) => Number(source.audio_ready_count || 0) > 0)
      .sort((a, b) => String(b.created_at || "").localeCompare(String(a.created_at || "")));
    for (const source of sources) {
      if (remaining <= 0) break;
      const input = Array.from(document.querySelectorAll(".generated-matrix-input")).find(
        (item) => item.dataset.genre === genre && item.dataset.sourceId === String(source.source_id || ""),
      );
      if (!input) continue;
      const max = Number.parseInt(input.max || "0", 10);
      const selected = Math.min(max, remaining);
      input.value = selected;
      remaining -= selected;
    }
  }
  renderGeneratedEvaluationPanel();
}

async function evaluateGeneratedResults() {
  const selections = readGeneratedEvaluationSelections();
  if (!selections.length) {
    throw new Error("Selecciona al menos una canción con WAV/MP3 por género/corrida.");
  }
  const metrics = ["fad", "kld", "tempo", "midi"];
  const includeClap = checked("#generatedMetricsUseClap");
  const genreSelections = {};
  for (const selection of selections) {
    genreSelections[selection.genre] = selection.rows;
  }
  const result = await postJob("/api/jobs/evaluation/from-results", {
    genre_selections: genreSelections,
    target_per_genre: Number.parseInt(valueOf("#generatedMetricsTarget") || "20", 10) || 20,
    pairing_strategy: "same_genre_round_robin",
    output_name: "resultados_generados",
    metrics,
    include_clap: includeClap,
  });
  renderJson(els.outputJson, result);
}


on(document.querySelector("#metricsRefreshButton"), "click", () => refreshAll().catch(showError));
on(document.querySelector("#generatedMetricsRefreshButton"), "click", () => refreshAll().catch(showError));
on(document.querySelector("#generatedMetricsRunButton"), "click", () => evaluateGeneratedResults().catch(showError));
on(document.querySelector("#generatedMetricsAutofillButton"), "click", () => setGeneratedMatrixSelection("target"));
on(document.querySelector("#generatedMetricsClearButton"), "click", () => setGeneratedMatrixSelection("clear"));
on(document.querySelector("#generatedMetricsTarget"), "input", updateGeneratedSelectionState);
document.addEventListener("input", (event) => {
  const target = event.target;
  if (!target) return;
  if (target.classList?.contains("genre-total-input")) {
    // Input simple por género: repartir entre corridas automáticamente.
    distributeGenreTotal(target.dataset.genre, target.value);
    updateGeneratedSelectionState();
  } else if (target.classList?.contains("generated-matrix-input")) {
    // Edición manual avanzada: reflejar la suma en el total del género.
    syncGenreTotalFromMatrix(target.dataset.genre);
    updateGeneratedSelectionState();
  }
});
on(document.querySelector("#metricsDistribute100Button"), "click", () => setMetricsDistribution("recommended"));
on(document.querySelector("#metricsUseMaxButton"), "click", () => setMetricsDistribution("max"));
on(document.querySelector("#metricsClearButton"), "click", () => setMetricsDistribution("clear"));
on(document.querySelector("#metricsGenerateBatchButton"), "click", () => generateEvaluationBatch().catch(showError));
on(document.querySelector("#metricsRunButton"), "click", () => runEvaluationMetrics().catch(showError));
on(document.querySelector("#metricsEvaluationSelect"), "change", renderEvaluationSummary);
on(document.querySelector("#generatedEvaluationSelect"), "change", renderGeneratedEvaluationSummary);
on(document.querySelector("#embeddingProjectionRunButton"), "click", () => runEmbeddingProjection());
on(document.querySelector("#embeddingProjectionRefreshButton"), "click", () => refreshAll().catch(showError));
document.querySelectorAll("#embeddingProjectionMethodToggle .method-tab").forEach((tab) =>
  on(tab, "click", () => setProjectionMethod(tab.dataset.projectionMethod)),
);
on(document.querySelector("#tokenTsneRunButton"), "click", () => runTokenTsne());
on(document.querySelector("#tokenTsneRefreshButton"), "click", () => runTokenTsne());

// Comparativa de modelos: solo se carga al entrar a la pestaña "Gráficas".
on(document.querySelector("#benchmarkRefreshButton"), "click", () => loadBenchmark(true));
on(document.querySelector("#benchmarkCloudButton"), "click", () => runCloudMetrics().catch(showError));
on(document.querySelector('.main-menu a[href="#graficas"]'), "click", () => loadBenchmark());
window.addEventListener("hashchange", () => {
  if (["#graficas", "#benchmark-charts"].includes(window.location.hash)) loadBenchmark();
});
document.addEventListener("change", (event) => {
  if (event.target?.classList?.contains("pair-metrics-control")) {
    renderGeneratedEvaluationSummary();
    renderEvaluationSummary();
  }
});

on(els.refreshButton, "click", () => refreshAll().catch(showError));
on(els.createProjectButton, "click", () => createProject().catch(showError));
on(els.cleanMidiButton, "click", () => cleanMidiDataset().catch(showError));
on(els.buildCatalogButton, "click", () => buildGenreCatalog().catch(showError));
on(els.tokenizeCatalogButton, "click", () => tokenizeCatalog().catch(showError));
on(els.exportInputTokensButton, "click", () => exportInputTokens().catch(showError));
on(els.exportOutputTokensButton, "click", () => exportOutputTokens().catch(showError));
on(els.trainTokenModelButton, "click", () => trainTokenModel().catch(showError));
on(els.generateTokensButton, "click", () => generateTokens().catch(showError));
on(els.generateRankedButton, "click", () => generateRanked().catch(showError));
on(els.renderMidiButton, "click", () => renderMidi().catch(showError));
on(els.renderLayersButton, "click", () => renderLayers().catch(showError));
on(els.analyzeMidiButton, "click", () => analyzeMidi().catch(showError));
on(els.blendEmbeddingsButton, "click", () => blendEmbeddings().catch(showError));
on(els.trainingPresetSelect, "change", applyTrainingPreset);
on(els.generationPresetSelect, "change", applyGenerationPreset);
on(els.toggleAdvanced, "change", (event) => {
  localStorage.setItem("hybrid_show_advanced", event.target.checked ? "1" : "0");
  applyAdvancedVisibility();
});
applyAdvancedVisibility();
document.addEventListener("change", (event) => {
  if (event.target?.id === "jamendoCatalogSelect") renderJamendoGenreOptions();
});
on(els.projectSelect, "change", async (event) => {
  state.selectedProjectId = event.target.value;
  await refreshSelectedProject();
  render();
});
document.addEventListener("click", (event) => {
  const renderButton = event.target.closest("button[data-render-candidate]");
  if (!renderButton) return;
  const midiPath = renderButton.dataset.renderCandidate;
  setSelectIfAvailable(els.generationSelect, midiPath);
  setValue("#renderName", `render_${midiPath.split("/").pop()?.replace(/\.mid$/i, "") || "candidate"}`);
  els.renderMidiButton?.scrollIntoView({ behavior: "smooth", block: "center" });
  els.jobStatus.textContent = "Listo para renderizar";
  els.jobMessage.textContent = "La candidata quedó seleccionada. Presiona “Renderizar audio” para crear WAV/MP3.";
});
on(els.steps, "click", async (event) => {
  const cycleButton = event.target.closest("button[data-cycle-action]");
  if (cycleButton) {
    try {
      const action = cycleButton.dataset.cycleAction;
      if (action === "clean-midi") await cleanMidiDataset();
      if (action === "build-catalog") await buildGenreCatalog();
      if (action === "augment-midi") await augmentMidiDataset();
      if (action === "tokenize-catalog") await tokenizeCatalog();
      if (action === "process-token-vae-demucs") await processJamendoClipsForTokenVae();
      if (action === "train-token-vae") await trainTokenVae();
      if (action === "encode-token-vae") await encodeTokenVae();
      if (action === "encode-genre-embeddings") await encodeGenreEmbeddings();
      if (action === "generate-genre-fusion") await generateWithGenreFusion();
      if (action === "blend-genre-embeddings") await blendGenreEmbeddings();
      if (action === "compare-fusions") await compareFusions();
      if (action === "generate-pretrained") await generatePretrainedModel();
      if (action === "connect-cloud") await connectCloud();
    } catch (error) {
      showError(error);
    }
    return;
  }
  const jamendoButton = event.target.closest("button[data-jamendo-action]");
  if (jamendoButton) {
    try {
      const action = jamendoButton.dataset.jamendoAction;
      if (action === "download") await downloadJamendo(true);
      if (action === "metadata") await downloadJamendo(false);
      if (action === "prepare") await prepareJamendoClips();
      if (action === "process") await processJamendoClips();
    } catch (error) {
      showError(error);
    }
    return;
  }
  const button = event.target.closest("button[data-step]");
  if (!button) return;
  const step = stepDefinitions.find((item) => item.id === button.dataset.step);
  if (!step) return;
  try {
    if (button.dataset.action === "secondary" && step.runSecondary) {
      await step.runSecondary();
    } else {
      await step.run();
    }
  } catch (error) {
    showError(error);
  }
});

function on(element, eventName, handler) {
  if (element) element.addEventListener(eventName, handler);
}

function showError(error) {
  els.jobStatus.textContent = "Error";
  els.jobMessage.textContent = error.message || String(error);
  els.jobProgress.style.width = "0";
}

// Model engine card selection (paso 4 · jamendo-train)
on(els.steps, "change", (event) => {
  const radio = event.target;
  if (radio.name !== "genEngine") return;
  const isCustom = radio.value === "custom";
  const customSection = document.getElementById("trainCustomSection");
  const pretrainedSection = document.getElementById("trainPretrainedSection");
  if (customSection) customSection.hidden = !isCustom;
  if (pretrainedSection) pretrainedSection.hidden = isCustom;
  document.querySelectorAll(".model-card[data-engine]").forEach((card) => {
    card.classList.toggle("is-selected", card.dataset.engine === radio.value);
  });
  const stepArticle = radio.closest("article[data-step-id='jamendo-train']");
  const btn = stepArticle?.querySelector("button[data-action='primary']");
  if (btn) {
    const engine = GEN_ENGINES.find((e) => e.value === radio.value);
    btn.textContent = isCustom ? "Entrenar modelo" : `Generar con ${engine?.name || radio.value}`;
  }
});

// Tema claro/oscuro (toggle en la esquina superior derecha).
function applyTheme(dark) {
  document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
  const btn = document.querySelector("#themeToggle");
  if (btn) btn.textContent = dark ? "☀️" : "🌙";
}
(function initTheme() {
  const saved = localStorage.getItem("hybrid_theme");
  const dark = saved
    ? saved === "dark"
    : Boolean(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
  applyTheme(dark);
})();
on(document.querySelector("#themeToggle"), "click", () => {
  const dark = document.documentElement.getAttribute("data-theme") !== "dark";
  localStorage.setItem("hybrid_theme", dark ? "dark" : "light");
  applyTheme(dark);
});

refreshAll().catch(showError);
