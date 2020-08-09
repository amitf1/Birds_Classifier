import wikipedia
import webbrowser


def open_wiki(spec):
    # for i, spec in enumerate(species):
    if spec == "BLACKBURNIAM WARBLER":
        spec = "BLACKBURNIAN WARBLER"
    elif spec == "CURL CRESTED ARACURI":
        spec = "CURL CRESTED ARACARI"
    elif spec == "EASTERN TOWEE":
        spec = "EASTERN TOWHEE"
    elif spec == "BROWN NOODY":
        spec = "BROWN NODDY"
    elif spec == "CANARY":
        spec = "DOMESTIC CANARY"
    elif spec == "CARMINE BEE-EATER":
        spec = "SOUTHERN CARMINE BEE-EATER"
    elif spec == "CHARA DE COLLAR":
        spec = "WOODHOUSE'S SCRUB JAY"
    elif spec == "MANDRIN DUCK":
        spec = "MANDARIN DUCK"
    elif spec == "KILLDEAR":
        spec = "KILLDEER"
    elif spec == "IMPERIAL SHAQ":
        spec = "IMPERIAL SHAG"
    elif spec == "PAINTED BUNTIG":
        spec = "PAINTED BUNTING"
    elif spec == "FRIGATE":
        spec = "FRIGATEBIRD"
    elif spec == "PURPLE SWAMPHEN":
        spec = "WESTERN SWAMPHEN"
    elif spec == "RED HONEY CREEPER":
        spec = "RED-LEGGED HONEYCREEPER"
    elif spec == "ROBIN":
        spec = "EUROPEAN ROBIN"
    elif spec == "RED WISKERED BULBUL":
        spec = "RED-WHISKERED BULBUL"
    elif spec == "RUFOUS KINGFISHER":
        spec = "ORIENTAL DWARF KINGFISHER"
    elif spec == "RUFUOS MOTMOT":
        spec = "RUFOUS MOTMOT"
    elif spec == "SORA":
        spec = "SORA (BIRD)"
    elif spec == "TEAL DUCK":
        spec = "EURASIAN TEAL"
    elif spec == "TIT MOUSE":
        spec = "TIT (BIRD)"
    elif spec == "TOUCHAN":
        spec = "TOUCAN"
    elif spec == "TOWNSENDS WARBLER":
        spec = "TOWNSEND'S WARBLER"
    elif spec == "TRUMPTER SWAN":
        spec = "TRUMPETER SWAN"
    elif spec == "VENEZUELIAN TROUPIAL":
        spec = "VENEZUELAN TROUPIAL"
    elif spec == "VERMILION FLYCATHER":
        spec = "VERMILION FLYCATCHER"

    search_res = wikipedia.search(spec)[0] if spec != "RUFOUS MOTMOT" else wikipedia.search(spec)[1]

    url = wikipedia.WikipediaPage(title=search_res).url
    webbrowser.open_new_tab(url)
