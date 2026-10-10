# Umber UI strings — pt-BR (Português do Brasil).
#
# Vocabulary follows pt-BR 3D-software usage, not literal translation:
# "Viewport", "Wireframe", "Bake" and "Assets" stay as the loanwords
# Brazilian artists use; "Malha" (mesh), "Camadas", "Grade" and "Grafo"
# are the established translations (Blender pt-BR). Labels are sentence
# case (pt-BR UI convention), not English title case. Keys missing here
# fall back to en-US.ftl.

## Barra de menus

menu =
    .file = Arquivo
    .view = Exibir
    .help = Ajuda
    .open-mesh = Abrir malha…
    .open-project = Abrir projeto…
    .save-project = Salvar projeto…
    .export-png = Exportar mapa pintado (PNG)…
    .load-environment = Carregar ambiente…
    .perf-hud = Mostrar HUD de desempenho
    .perf-hud-unavailable = O HUD de desempenho requer --features perf
    .wireframe = Mostrar wireframe (W)
    .grid = Mostrar grade (G)
    .about = Sobre o Umber
    .about-text = Umber v0.1.0 — Onda 2 em andamento

## Títulos dos painéis

panel =
    .viewport = Viewport
    .uv-view = UV 2D
    .layers = Camadas
    .properties = Propriedades
    .assets = Assets
    .history = Histórico
    .texture-sets = Conjuntos de texturas
    .bakes = Bakes
    .export = Exportação
    .graph = Grafo
    .display = Exibição

## Botões principais

button =
    .bake = Fazer bake
    .export = Exportar
    .evaluate = Avaliar
    .save = Salvar
    .save-as = Salvar como…
    .reset = Redefinir
    .choose = Escolher…
    .add = Adicionar

## Mensagens de status

status =
    .baking = { $count ->
        [one] Fazendo bake de { $count } mapa…
       *[other] Fazendo bake de { $count } mapas…
    }
    .plugins = { $loaded ->
        [one] { $loaded } plugin carregado
       *[other] { $loaded } plugins carregados
    }, { $failed } com falha

## Configurações

settings =
    .language = Idioma

## Fragmentos compartilhados

common =
    .none = nenhum
    .out-dir = Saída: { $dir }

## Shell do app: painéis provisórios, filtros dos diálogos de arquivo

shell =
    .assets-placeholder = Assets / prateleira (Onda 4+)
    .texture-sets-placeholder = Conjuntos de texturas (Onda 2)
    .filter-environment = Ambiente
    .filter-meshes = Malhas

## Painel de bakes

bakes =
    .map-ao = Oclusão de ambiente
    .map-curvature = Curvatura
    .map-thickness = Espessura
    .map-position = Posição
    .no-mesh = Nenhuma malha carregada — abra uma malha para fazer bake.
    .no-gpu = Sem dispositivo de GPU — o bake precisa do dispositivo wgpu.
    .resolution = Resolução
    .rays = Raios
    .dilation = Dilatação
    .select-map = Selecione ao menos um mapa para o bake.
    .select-tile = Selecione ao menos um tile para o bake.
    .no-bake-yet = Nenhum bake ainda.
    .done = { $count ->
        [one] Bake de { $count } mapa concluído em { $ms } ms: { $files }
       *[other] Bake de { $count } mapas concluído em { $ms } ms: { $files }
    }
    .skipped-note = ignorados (a v1 só faz bake de AO por tile; os demais mapas só no tile 1001): { $pairs }
    .failed = Falha no bake: { $error }
    .worker-failed = Falha no bake: não foi possível iniciar o worker de bake: { $error }

## Painel de exportação

export =
    .no-mesh = Nenhuma malha carregada — abra uma malha para exportar.
    .no-gpu = Sem dispositivo de GPU — a exportação precisa do dispositivo wgpu.
    .preset = Predefinição
    .size = Tamanho
    .materialx = MaterialX (.mtlx)
    .select-tile = Selecione ao menos um tile para exportar.
    .tile-badge = { $count ->
        [one] { $count } tile
       *[other] { $count } tiles
    }
    .source-painted = Fonte da Base Color: pintura{ $tiles } — o que você pintou é o que será exportado.
    .source-graph = Fonte da Base Color: nó { $node } do grafo{ $tiles } — a saída avaliada do painel.
    .source-flat-size = Fonte da Base Color: cor sólida provisória — o alvo de pintura é { $width } × { $height }, não o tamanho de exportação { $size } × { $size }.
    .source-flat-none = Fonte da Base Color: cor sólida provisória — nenhuma sessão de pintura ou saída de grafo ativa.
    .base-painted = pintura
    .base-graph = nó { $node } do grafo
    .base-flat = cor sólida provisória
    .base-none = Base Color: nenhuma
    .base = Base Color: { $source }
    .base-tiles = Base Color: { $source } ({ $tiles })
    .no-export-yet = Nenhuma exportação ainda.
    .done = { $count ->
        [one] { $count } saída exportada ({ $base }; bake { $bake-ms } ms, gravação { $write-ms } ms): { $files }
       *[other] { $count } saídas exportadas ({ $base }; bake { $bake-ms } ms, gravação { $write-ms } ms): { $files }
    }
    .skipped-output = { $file } (faltando { $maps })
    .skipped = ignoradas: { $outputs }
    .tiles-skipped = tiles ignorados (sem geometria de malha): { $tiles }
    .failed = Falha na exportação: { $error }

## Painel do grafo (lista, editor, linha de status)

graph =
    .view-canvas = Canvas
    .view-list = Lista
    .canvas-help = Arraste o espaço vazio para mover a vista, use a roda para zoom, arraste uma porta de saída até uma entrada para conectar, clique com o botão direito para adicionar; Delete remove a aresta selecionada.
    .output = Saída: { $node }
    .nodes = Nós
    .no-nodes = Nenhum nó ainda — adicione um abaixo.
    .add-node = Adicionar nó
    .remove-selected = Remover selecionado
    .select-node = Selecione um nó para editar seus parâmetros.
    .node-title = Nó { $id }: { $def }
    .set-output = Definir como saída
    .edges = Arestas
    .edge = { $input } ← nó { $from }
    .edge-input = entrada
    .edge-from = de
    .add-edge = Adicionar aresta
    .no-output = Nenhuma saída ainda — clique em Avaliar.
    .uniform-output = Saída uniforme: { $value }
    .no-eval-yet = Nenhuma avaliação ainda.
    .loaded = Grafo carregado — clique em Avaliar.
    .loaded-unknown = { $count ->
        [one] Grafo carregado com { $count } tipo de nó desconhecido — clique em Avaliar.
       *[other] Grafo carregado com { $count } tipos de nó desconhecidos — clique em Avaliar.
    }
    .load-failed = Falha ao carregar o grafo: { $error }
    .empty = O grafo está vazio — adicione um nó primeiro.
    .no-output-node = Nenhum nó de saída — selecione um primeiro.
    .evaluated = { $count ->
        [one] { $count } nó avaliado em { $width }x{ $height }.
       *[other] { $count } nós avaliados em { $width }x{ $height }.
    }
    .eval-failed = Falha na avaliação: { $error }

## Canvas do grafo (editor de nós)

canvas =
    .empty-hint = Clique com o botão direito para adicionar um nó
    .add-node = Adicionar nó

## Painel de exibição

display =
    .view = Vista
    .exposure = Exposição (EV)
    .gamma = Gama
    .preview = Prévia (rampa linear, 1 EV por amostra):
    .chain-identity = identidade (Raw, 0 EV, gama 1)
    .chain = exposição { $exposure } EV → { $view } → gama { $gamma }
    .live = Ativa no viewport 3D (passe da malha) e na vista UV (exibição da pintura) pela LUT de exibição da GPU.
    .rebuilding = Reconstruindo a LUT de exibição da GPU para o viewport 3D e a vista UV.
    .active-chain = Cadeia ativa: { $chain }. { $status }

## Painel de propriedades do pincel

brush =
    .no-preset = Nenhuma predefinição selecionada
    .preset = Predefinição
    .preset-none = Nenhuma
    .broken = { $count ->
        [one] { $count } com defeito
       *[other] { $count } com defeito
    }
    .overridden = { $count ->
        [one] { $count } substituída
       *[other] { $count } substituídas
    }
    .name = Nome
    .color = Cor
    .alpha = Opacidade
    .hardness = Dureza
    .pressure-gamma = Gama da pressão
    .stabilizer = Estabilizador (one-euro)
    .min-cutoff = Corte mínimo
    .beta = Beta
    .d-cutoff = Corte d
    .lazy-mouse = Lazy mouse
    .radius-px = Raio (px)
    .strength = Intensidade
    .spacing = Espaçamento
    .dabs-per-radius = Dabs por raio
    .pressure-alpha = Pressão → opacidade
    .pressure-radius = Pressão → raio
    .save-as-label = Salvar como
    .save-as-hint = Nome da nova predefinição
    .save-failed = Falha ao salvar: { $error }
    .save-as-needs-name = Salvar como precisa de um nome
    .save-as-needs-preset = Salvar como precisa de uma predefinição ativa
    .save-as-no-dir = Falha em Salvar como: sem pasta de predefinições do usuário
    .save-as-failed = Falha em Salvar como: { $error }
    .reset-failed = Falha ao redefinir: { $error }

## Painel de camadas

layers =
    .add-paint = ＋ Adicionar camada de pintura
    .default-paint-name = Pintura { $n }
    .kind-paint = pintura
    .kind-fill = preenchimento
    .kind-folder = pasta

## Painel de histórico

history =
    .undo = ↶ Desfazer
    .redo = ↷ Refazer
    .journal = { $count ->
        [one] { $count } entrada no histórico
       *[other] { $count } entradas no histórico
    }

## Seletor de tiles UDIM (Bakes + Exportação)

tiles =
    .label = Tiles:
    .tile = Tile { $tile }

## Combo de tamanho (Bakes + Exportação)

size =
    .vram-note = { $size } × { $size } (~{ $mib } MB/buffer de texels)

## Vista UV 2D

uv =
    .no-mesh = nenhuma malha carregada
