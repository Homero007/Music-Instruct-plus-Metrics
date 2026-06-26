const API_BASE = localStorage.getItem("hybrid_api_base") || "http://127.0.0.1:8100";

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
  jobMessage: document.querySelector("#jobMessage"),
  jobEvents: document.querySelector("#jobEvents"),
  steps: document.querySelector("#steps"),
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

const stepDefinitions = [
  {
    id: "import",
    title: "Importar audio",
    help: "Elige el material base: un audio individual o pistas descargadas de MTG-Jamendo para entrenamiento.",
    requirement: () => Boolean(state.selectedProject),
    complete: () =>
      Boolean(state.selectedProject?.source?.normalized || localStorage.getItem("hybrid_selected_jamendo_catalog_path")),
    actionLabel: "Importar audio individual",
    secondaryLabel: "Usar pistas seleccionadas",
    renderFields: () => `
      <label class="field">
        <span>Audio individual</span>
        <input id="audioPath" placeholder="/Users/tu_usuario/Music/cancion.wav" />
        <small>Opcional. Úsalo si quieres analizar una canción concreta.</small>
      </label>
      <div class="selector-box">
        <div class="panel-heading compact-heading">
          <span>Pistas descargadas</span>
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
          <summary>Descargar más pistas o preparar clips</summary>
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
            <small>Prioriza manifests creados con “Procesar con Demucs para Token-VAE”.</small>
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
    `,
    run: () => trainJamendoModel(),
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

function renderAudioPlayers() {
  if (!els.audioPlayers) return;
  const playable = [];
  for (const render of state.resources.renders || []) {
    const url = render.mp3_download_url || render.wav_download_url;
    if (url) {
      playable.push({
        label: `${render.generation_mode_label || "Render de audio"} · ${render.label}`,
        mode: render.generation_mode || "render",
        url,
        wavUrl: render.wav_download_url,
        mp3Url: render.mp3_download_url,
      });
    }
  }
  for (const ranking of state.resources.rankings || []) {
    for (const candidate of ranking.candidates || []) {
      const url = candidate.mp3_download_url || candidate.wav_download_url;
      if (url) {
        playable.push({
          label: `${ranking.generation_mode_label || "Generación"} · ${ranking.ranked_id} · ${candidate.candidate_id} · score ${formatNumber(candidate.score)}`,
          mode: ranking.generation_mode || "transformer",
          url,
          wavUrl: candidate.wav_download_url,
          mp3Url: candidate.mp3_download_url,
          midiUrl: candidate.midi_download_url,
        });
      }
    }
  }
  els.audioPlayers.innerHTML = playable.length
    ? playable
        .slice(-10)
        .reverse()
        .map(
          (item) => `
            <div class="audio-player">
              <strong>
                <span class="mode-badge ${escapeHtml(item.mode || "render")}">${escapeHtml(item.label.split(" · ")[0] || "Render")}</span>
                ${escapeHtml(item.label.split(" · ").slice(1).join(" · ") || item.label)}
              </strong>
              <audio controls src="${API_BASE}${encodeURI(item.url)}"></audio>
              <span class="file-links">
                ${item.midiUrl ? `<a href="${API_BASE}${encodeURI(item.midiUrl)}" download>MIDI</a>` : ""}
                ${item.wavUrl ? `<a href="${API_BASE}${encodeURI(item.wavUrl)}" download>WAV</a>` : ""}
                ${item.mp3Url ? `<a href="${API_BASE}${encodeURI(item.mp3Url)}" download>MP3</a>` : ""}
              </span>
            </div>
          `,
        )
        .join("")
    : `<small class="muted">Cuando renderices WAV/MP3 podrás escucharlos aquí.</small>`;
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
  els.steps.innerHTML = activeStepDefinitions()
    .map((step, index) => {
      const ready = step.requirement();
      const complete = step.complete();
      const disabled = !ready || state.busy;
      const statusClass = complete ? "" : ready ? "warn" : "error";
      const statusText = complete ? "Listo" : ready ? "Pendiente" : "Bloqueado";
      return `
        <article class="step">
          <div>
            <div class="step-header">
              <span class="step-number">Paso ${index + 1} · <b class="status-pill ${statusClass}">${statusText}</b></span>
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
}

function activeStepDefinitions() {
  const jamendoMode = isJamendoMode();
  return stepDefinitions.filter((step) => {
    if (step.mode === "jamendo") return jamendoMode;
    if (jamendoMode && ["stems", "midi", "features", "vae", "embedding"].includes(step.id)) {
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
  const events = (job.events || []).slice(-8).reverse();
  els.jobEvents.innerHTML = events
    .map((event) => `<li>${escapeHtml(event.time || "")} · ${escapeHtml(event.stage || "")}<br />${escapeHtml(event.message || "")}</li>`)
    .join("");
  if (job.status === "completed" && job.result) {
    if (["train-token-model", "generate-tokens", "generate-ranked", "train-token-vae", "encode-token-vae", "encode-genre-embeddings"].includes(job.kind)) {
      renderJson(els.modelJson, job.result);
    } else if (["render-midi", "render-layers", "midi-metrics"].includes(job.kind)) {
      renderJson(els.renderJson, job.result);
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

refreshAll().catch(showError);
