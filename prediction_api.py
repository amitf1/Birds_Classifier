from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import load_model
import pathlib
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import wikipedia
import webbrowser

# MODEL = load_model('model2')
UPLOAD_FOLDER = 'data/Predictions/'
MODEL = load_model('new_model.h5')

TRAIN_DIR = 'data/200_species_train'
CATEGORIES = ['OSTRICH', 'VIOLET GREEN SWALLOW', 'WHITE TAILED TROPIC', 'COUCHS KINGBIRD',
                       'PURPLE GALLINULE', 'RAZORBILL', 'VENEZUELIAN TROUPIAL', 'TREE SWALLOW', 'WHITE NECKED RAVEN',
                       'MANDRIN DUCK', 'TOUCHAN', 'WILSONS BIRD OF PARADISE', 'EMPEROR PENGUIN', 'LILAC ROLLER',
                       'BAY-BREASTED WARBLER', 'SPLENDID WREN', 'WILD TURKEY', 'CASSOWARY', 'RUFUOS MOTMOT',
                       'NORTHERN JACANA', 'BLACK VULTURE', 'AMERICAN REDSTART', 'ROADRUNNER', 'BARN SWALLOW',
                       'FLAMINGO', 'HOODED MERGANSER', 'SORA', 'HOUSE SPARROW', 'BIRD OF PARADISE', 'AMERICAN KESTREL',
                       'EURASIAN MAGPIE', 'PEREGRINE FALCON', 'SCARLET MACAW', 'AFRICAN FIREFINCH', 'TOWNSENDS WARBLER',
                       'GOULDIAN FINCH', 'RUBY THROATED HUMMINGBIRD', 'WOOD DUCK', 'GLOSSY IBIS', 'EASTERN MEADOWLARK',
                       'EASTERN BLUEBIRD', 'INDIGO BUNTING', 'COCK OF THE  ROCK', 'GOLDEN CHEEKED WARBLER',
                       'MIKADO  PHEASANT', 'MASKED BOOBY', 'BELTED KINGFISHER', 'CHARA DE COLLAR', 'JAVAN MAGPIE',
                       'BROWN NOODY', 'GUINEAFOWL', 'HORNBILL', 'RED WINGED BLACKBIRD', 'GRAY PARTRIDGE', 'BANANAQUIT',
                       'HOUSE FINCH', 'BLACKBURNIAM WARBLER', 'PURPLE MARTIN', 'CUBAN TODY', 'NORTHERN RED BISHOP',
                       'AMERICAN PIPIT', 'ROCK DOVE', 'BALTIMORE ORIOLE', 'DOWNY WOODPECKER', 'BLACK-NECKED GREBE',
                       'COMMON POORWILL', 'CANARY', 'CALIFORNIA QUAIL', 'RED WISKERED BULBUL', 'CARMINE BEE-EATER',
                       'SPOONBILL', 'MALEO', 'BLACK FRANCOLIN', 'PUFFIN', 'GOLDEN PHEASANT', 'ROBIN', 'BOBOLINK',
                       'ELLIOTS  PHEASANT', 'EASTERN ROSELLA', 'PEACOCK', 'NICOBAR PIGEON', 'EASTERN TOWEE',
                       'LONG-EARED OWL', 'BEARDED BARBET', 'CALIFORNIA GULL', 'BLUE HERON', 'ALBATROSS', 'SHOEBILL',
                       'AMERICAN AVOCET', 'COMMON GRACKLE', 'CAPUCHINBIRD', 'AMERICAN BITTERN', 'PURPLE SWAMPHEN',
                       'RING-BILLED GULL', 'BLACK SWAN', 'YELLOW HEADED BLACKBIRD', 'RAINBOW LORIKEET',
                       'BLACK-THROATED SPARROW', 'COCKATOO', 'CROW', 'PARADISE TANAGER', 'IMPERIAL SHAQ',
                       'PURPLE FINCH', 'BLACK-CAPPED CHICKADEE', 'AMERICAN COOT', 'KILLDEAR', 'EMU', 'CRESTED AUKLET',
                       'STRAWBERRY FINCH', 'TIT MOUSE', 'MARABOU STORK', 'MALLARD DUCK', 'MALACHITE KINGFISHER',
                       'SPANGLED COTINGA', 'BARN OWL', 'STEAMER DUCK', 'MYNA', 'DARK EYED JUNCO', 'TURKEY VULTURE',
                       'NORTHERN GANNET', 'NORTHERN GOSHAWK', 'GOLDEN CHLOROPHONIA', 'WHITE CHEEKED TURACO',
                       'PARUS MAJOR', 'COMMON HOUSE MARTIN', 'HOOPOES', 'ANTBIRD', 'CRESTED CARACARA',
                       'BAR-TAILED GODWIT', 'CURL CRESTED ARACURI', 'VERMILION FLYCATHER', 'RED HEADED WOODPECKER',
                       'BALD EAGLE', 'VARIED THRUSH', 'SAND MARTIN', 'BLUE GROUSE', 'ARARIPE MANAKIN',
                       'CHIPPING SPARROW', 'RED FACED CORMORANT', 'CROWNED PIGEON', 'ALEXANDRINE PARAKEET',
                       'AMERICAN GOLDFINCH', 'KING VULTURE', 'RUFOUS KINGFISHER', 'GRAY CATBIRD', 'LARK BUNTING',
                       'ELEGANT TROGON', 'PAINTED BUNTIG', 'GILA WOODPECKER', 'CASPIAN TERN', 'JABIRU', 'INCA TERN',
                       'OSPREY', 'COMMON STARLING', 'ANNAS HUMMINGBIRD', 'FLAME TANAGER', 'MOURNING DOVE',
                       'PINK ROBIN', 'RING-NECKED PHEASANT', 'ANHINGA', 'SNOWY OWL', 'BLACK SKIMMER', 'HYACINTH MACAW',
                       'GOLD WING WARBLER', 'HAWAIIAN GOOSE', 'CALIFORNIA CONDOR', 'GILDED FLICKER', 'CINNAMON TEAL',
                       'NORTHERN MOCKINGBIRD', 'OCELLATED TURKEY', 'WATTLED CURASSOW', 'RED THROATED BEE EATER',
                       'NORTHERN PARULA', 'BROWN THRASHER', 'RED FACED WARBLER', 'GREEN JAY', 'COMMON LOON', 'FRIGATE',
                       'D-ARNAUDS BARBET', 'NORTHERN CARDINAL', 'TURQUOISE MOTMOT', 'STORK BILLED KINGFISHER',
                       'TEAL DUCK', 'ROSY FACED LOVEBIRD', 'SNOWY EGRET', 'TRUMPTER SWAN', 'CACTUS WREN',
                       'SCARLET IBIS', 'BLACK THROATED WARBLER', 'QUETZAL', 'EVENING GROSBEAK', 'NORTHERN FLICKER',
                       'TAIWAN MAGPIE', 'GREY PLOVER', 'ROUGH LEG BUZZARD', 'CAPE MAY WARBLER', 'RED HEADED DUCK',
                       'GOLDEN EAGLE', 'RED HONEY CREEPER', 'PELICAN']
CATEGORIES = np.array(sorted(CATEGORIES))
encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy', allow_pickle=True)


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



def getPrediction(filename):
    model = MODEL
    image = load_img(UPLOAD_FOLDER+filename, target_size=(224, 224), interpolation='bicubic')
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # image = preprocess_input(image)
    image = image/255.0
    pred = model.predict(image)
    label = pred.argmax(axis=1)
    specie = encoder.inverse_transform(label)[0].decode('ascii')
    # specie = CATEGORIES[label][0]
    print(label, specie, pred[0, label][0]*100)
    open_wiki(specie)

    return specie, pred[0, label][0]*100
