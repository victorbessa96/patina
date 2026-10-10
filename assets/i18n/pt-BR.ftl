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
