"""Create tools for Pytorch to load Downsampled ImageNet (32X32,64X64)

Thanks to the cifar.py provided by Pytorch.

Author: Xu Ma.
Date:   Apr/21/2019

Data Preparation:
    1. Download unsampled data from ImageNet website.
    2. Unzip file  to rootPath. eg: /home/xm0036/Datasets/ImageNet64(no train, val folders)

Remark:
This tool is able to automatic recognize downsampled size.


Use this tool like cifar10 in datsets/torchvision.
"""


from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data


class ImageNetDownSample(data.Dataset):
    """`DownsampleImageNet`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    train_list = [
        ['train_data_batch_1'],
        ['train_data_batch_2'],
        ['train_data_batch_3'],
        ['train_data_batch_4'],
        ['train_data_batch_5'],
        ['train_data_batch_6'],
        ['train_data_batch_7'],
        ['train_data_batch_8'],
        ['train_data_batch_9'],
        ['train_data_batch_10']
    ]
    test_list = [
        ['val_data'],
    ]

    def __init__(self, root, train=True,
                 transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.classes = ['kit_fox', 'English_setter', 'Siberian_husky', 'Australian_terrier', 'English_springer', 'grey_whale', 'lesser_panda', 'Egyptian_cat', 'ibex', 'Persian_cat', 'cougar', 'gazelle', 'porcupine', 'sea_lion', 'malamute', 'badger', 'Great_Dane', 'Walker_hound', 'Welsh_springer_spaniel', 'whippet', 'Scottish_deerhound', 'killer_whale', 'mink', 'African_elephant', 'Weimaraner', 'soft-coated_wheaten_terrier', 'Dandie_Dinmont', 'red_wolf', 'Old_English_sheepdog', 'jaguar', 'otterhound', 'bloodhound', 'Airedale', 'hyena', 'meerkat', 'giant_schnauzer', 'titi', 'three-toed_sloth', 'sorrel', 'black-footed_ferret', 'dalmatian', 'black-and-tan_coonhound', 'papillon', 'skunk', 'Staffordshire_bullterrier', 'Mexican_hairless', 'Bouvier_des_Flandres', 'weasel', 'miniature_poodle', 'Cardigan', 'malinois', 'bighorn', 'fox_squirrel', 'colobus', 'tiger_cat', 'Lhasa', 'impala', 'coyote', 'Yorkshire_terrier', 'Newfoundland', 'brown_bear', 'red_fox', 'Norwegian_elkhound', 'Rottweiler', 'hartebeest', 'Saluki', 'grey_fox', 'schipperke', 'Pekinese', 'Brabancon_griffon', 'West_Highland_white_terrier', 'Sealyham_terrier', 'guenon', 'mongoose', 'indri', 'tiger', 'Irish_wolfhound', 'wild_boar', 'EntleBucher', 'zebra', 'ram', 'French_bulldog', 'orangutan', 'basenji', 'leopard', 'Bernese_mountain_dog', 'Maltese_dog', 'Norfolk_terrier', 'toy_terrier', 'vizsla', 'cairn', 'squirrel_monkey', 'groenendael', 'clumber', 'Siamese_cat', 'chimpanzee', 'komondor', 'Afghan_hound', 'Japanese_spaniel', 'proboscis_monkey', 'guinea_pig', 'white_wolf', 'ice_bear', 'gorilla', 'borzoi', 'toy_poodle', 'Kerry_blue_terrier', 'ox', 'Scotch_terrier', 'Tibetan_mastiff', 'spider_monkey', 'Doberman', 'Boston_bull', 'Greater_Swiss_Mountain_dog', 'Appenzeller', 'Shih-Tzu', 'Irish_water_spaniel', 'Pomeranian', 'Bedlington_terrier', 'warthog', 'Arabian_camel', 'siamang', 'miniature_schnauzer', 'collie', 'golden_retriever', 'Irish_terrier', 'affenpinscher', 'Border_collie', 'hare', 'boxer', 'silky_terrier', 'beagle', 'Leonberg', 'German_short-haired_pointer', 'patas', 'dhole', 'baboon', 'macaque', 'Chesapeake_Bay_retriever', 'bull_mastiff', 'kuvasz', 'capuchin', 'pug', 'curly-coated_retriever', 'Norwich_terrier', 'flat-coated_retriever', 'hog', 'keeshond', 'Eskimo_dog', 'Brittany_spaniel', 'standard_poodle', 'Lakeland_terrier', 'snow_leopard', 'Gordon_setter', 'dingo', 'standard_schnauzer', 'hamster', 'Tibetan_terrier', 'Arctic_fox', 'wire-haired_fox_terrier', 'basset', 'water_buffalo', 'American_black_bear', 'Angora', 'bison', 'howler_monkey', 'hippopotamus', 'chow', 'giant_panda', 'American_Staffordshire_terrier', 'Shetland_sheepdog', 'Great_Pyrenees', 'Chihuahua', 'tabby', 'marmoset', 'Labrador_retriever', 'Saint_Bernard', 'armadillo', 'Samoyed', 'bluetick', 'redbone', 'polecat', 'marmot', 'kelpie', 'gibbon', 'llama', 'miniature_pinscher', 'wood_rabbit', 'Italian_greyhound', 'lion', 'cocker_spaniel', 'Irish_setter', 'dugong', 'Indian_elephant', 'beaver', 'Sussex_spaniel', 'Pembroke', 'Blenheim_spaniel', 'Madagascar_cat', 'Rhodesian_ridgeback', 'lynx', 'African_hunting_dog', 'langur', 'Ibizan_hound', 'timber_wolf', 'cheetah', 'English_foxhound', 'briard', 'sloth_bear', 'Border_terrier', 'German_shepherd', 'otter', 'koala', 'tusker', 'echidna', 'wallaby', 'platypus', 'wombat', 'revolver', 'umbrella', 'schooner', 'soccer_ball', 'accordion', 'ant', 'starfish', 'chambered_nautilus', 'grand_piano', 'laptop', 'strawberry', 'airliner', 'warplane', 'airship', 'balloon', 'space_shuttle', 'fireboat', 'gondola', 'speedboat', 'lifeboat', 'canoe', 'yawl', 'catamaran', 'trimaran', 'container_ship', 'liner', 'pirate', 'aircraft_carrier', 'submarine', 'wreck', 'half_track', 'tank', 'missile', 'bobsled', 'dogsled', 'bicycle-built-for-two', 'mountain_bike', 'freight_car', 'passenger_car', 'barrow', 'shopping_cart', 'motor_scooter', 'forklift', 'electric_locomotive', 'steam_locomotive', 'amphibian', 'ambulance', 'beach_wagon', 'cab', 'convertible', 'jeep', 'limousine', 'minivan', 'Model_T', 'racer', 'sports_car', 'go-kart', 'golfcart', 'moped', 'snowplow', 'fire_engine', 'garbage_truck', 'pickup', 'tow_truck', 'trailer_truck', 'moving_van', 'police_van', 'recreational_vehicle', 'streetcar', 'snowmobile', 'tractor', 'mobile_home', 'tricycle', 'unicycle', 'horse_cart', 'jinrikisha', 'oxcart', 'bassinet', 'cradle', 'crib', 'four-poster', 'bookcase', 'china_cabinet', 'medicine_chest', 'chiffonier', 'table_lamp', 'file', 'park_bench', 'barber_chair', 'throne', 'folding_chair', 'rocking_chair', 'studio_couch', 'toilet_seat', 'desk', 'pool_table', 'dining_table', 'entertainment_center', 'wardrobe', 'Granny_Smith', 'orange', 'lemon', 'fig', 'pineapple', 'banana', 'jackfruit', 'custard_apple', 'pomegranate', 'acorn', 'hip', 'ear', 'rapeseed', 'corn', 'buckeye', 'organ', 'upright', 'chime', 'drum', 'gong', 'maraca', 'marimba', 'steel_drum', 'banjo', 'cello', 'violin', 'harp', 'acoustic_guitar', 'electric_guitar', 'cornet', 'French_horn', 'trombone', 'harmonica', 'ocarina', 'panpipe', 'bassoon', 'oboe', 'sax', 'flute', 'daisy', "yellow_lady's_slipper", 'cliff', 'valley', 'alp', 'volcano', 'promontory', 'sandbar', 'coral_reef', 'lakeside', 'seashore', 'geyser', 'hatchet', 'cleaver', 'letter_opener', 'plane', 'power_drill', 'lawn_mower', 'hammer', 'corkscrew', 'can_opener', 'plunger', 'screwdriver', 'shovel', 'plow', 'chain_saw', 'cock', 'hen', 'ostrich', 'brambling', 'goldfinch', 'house_finch', 'junco', 'indigo_bunting', 'robin', 'bulbul', 'jay', 'magpie', 'chickadee', 'water_ouzel', 'kite', 'bald_eagle', 'vulture', 'great_grey_owl', 'black_grouse', 'ptarmigan', 'ruffed_grouse', 'prairie_chicken', 'peacock', 'quail', 'partridge', 'African_grey', 'macaw', 'sulphur-crested_cockatoo', 'lorikeet', 'coucal', 'bee_eater', 'hornbill', 'hummingbird', 'jacamar', 'toucan', 'drake', 'red-breasted_merganser', 'goose', 'black_swan', 'white_stork', 'black_stork', 'spoonbill', 'flamingo', 'American_egret', 'little_blue_heron', 'bittern', 'crane', 'limpkin', 'American_coot', 'bustard', 'ruddy_turnstone', 'red-backed_sandpiper', 'redshank', 'dowitcher', 'oystercatcher', 'European_gallinule', 'pelican', 'king_penguin', 'albatross', 'great_white_shark', 'tiger_shark', 'hammerhead', 'electric_ray', 'stingray', 'barracouta', 'coho', 'tench', 'goldfish', 'eel', 'rock_beauty', 'anemone_fish', 'lionfish', 'puffer', 'sturgeon', 'gar', 'loggerhead', 'leatherback_turtle', 'mud_turtle', 'terrapin', 'box_turtle', 'banded_gecko', 'common_iguana', 'American_chameleon', 'whiptail', 'agama', 'frilled_lizard', 'alligator_lizard', 'Gila_monster', 'green_lizard', 'African_chameleon', 'Komodo_dragon', 'triceratops', 'African_crocodile', 'American_alligator', 'thunder_snake', 'ringneck_snake', 'hognose_snake', 'green_snake', 'king_snake', 'garter_snake', 'water_snake', 'vine_snake', 'night_snake', 'boa_constrictor', 'rock_python', 'Indian_cobra', 'green_mamba', 'sea_snake', 'horned_viper', 'diamondback', 'sidewinder', 'European_fire_salamander', 'common_newt', 'eft', 'spotted_salamander', 'axolotl', 'bullfrog', 'tree_frog', 'tailed_frog', 'whistle', 'wing', 'paintbrush', 'hand_blower', 'oxygen_mask', 'snorkel', 'loudspeaker', 'microphone', 'screen', 'mouse', 'electric_fan', 'oil_filter', 'strainer', 'space_heater', 'stove', 'guillotine', 'barometer', 'rule', 'odometer', 'scale', 'analog_clock', 'digital_clock', 'wall_clock', 'hourglass', 'sundial', 'parking_meter', 'stopwatch', 'digital_watch', 'stethoscope', 'syringe', 'magnetic_compass', 'binoculars', 'projector', 'sunglasses', 'loupe', 'radio_telescope', 'bow', 'cannon', 'assault_rifle', 'rifle', 'projectile', 'computer_keyboard', 'typewriter_keyboard', 'crane', 'lighter', 'abacus', 'cash_machine', 'slide_rule', 'desktop_computer', 'hand-held_computer', 'notebook', 'web_site', 'harvester', 'thresher', 'printer', 'slot', 'vending_machine', 'sewing_machine', 'joystick', 'switch', 'hook', 'car_wheel', 'paddlewheel', 'pinwheel', "potter's_wheel", 'gas_pump', 'carousel', 'swing', 'reel', 'radiator', 'puck', 'hard_disc', 'sunglass', 'pick', 'car_mirror', 'solar_dish', 'remote_control', 'disk_brake', 'buckle', 'hair_slide', 'knot', 'combination_lock', 'padlock', 'nail', 'safety_pin', 'screw', 'muzzle', 'seat_belt', 'ski', 'candle', "jack-o'-lantern", 'spotlight', 'torch', 'neck_brace', 'pier', 'tripod', 'maypole', 'mousetrap', 'spider_web', 'trilobite', 'harvestman', 'scorpion', 'black_and_gold_garden_spider', 'barn_spider', 'garden_spider', 'black_widow', 'tarantula', 'wolf_spider', 'tick', 'centipede', 'isopod', 'Dungeness_crab', 'rock_crab', 'fiddler_crab', 'king_crab', 'American_lobster', 'spiny_lobster', 'crayfish', 'hermit_crab', 'tiger_beetle', 'ladybug', 'ground_beetle', 'long-horned_beetle', 'leaf_beetle', 'dung_beetle', 'rhinoceros_beetle', 'weevil', 'fly', 'bee', 'grasshopper', 'cricket', 'walking_stick', 'cockroach', 'mantis', 'cicada', 'leafhopper', 'lacewing', 'dragonfly', 'damselfly', 'admiral', 'ringlet', 'monarch', 'cabbage_butterfly', 'sulphur_butterfly', 'lycaenid', 'jellyfish', 'sea_anemone', 'brain_coral', 'flatworm', 'nematode', 'conch', 'snail', 'slug', 'sea_slug', 'chiton', 'sea_urchin', 'sea_cucumber', 'iron', 'espresso_maker', 'microwave', 'Dutch_oven', 'rotisserie', 'toaster', 'waffle_iron', 'vacuum', 'dishwasher', 'refrigerator', 'washer', 'Crock_Pot', 'frying_pan', 'wok', 'caldron', 'coffeepot', 'teapot', 'spatula', 'altar', 'triumphal_arch', 'patio', 'steel_arch_bridge', 'suspension_bridge', 'viaduct', 'barn', 'greenhouse', 'palace', 'monastery', 'library', 'apiary', 'boathouse', 'church', 'mosque', 'stupa', 'planetarium', 'restaurant', 'cinema', 'home_theater', 'lumbermill', 'coil', 'obelisk', 'totem_pole', 'castle', 'prison', 'grocery_store', 'bakery', 'barbershop', 'bookshop', 'butcher_shop', 'confectionery', 'shoe_shop', 'tobacco_shop', 'toyshop', 'fountain', 'cliff_dwelling', 'yurt', 'dock', 'brass', 'megalith', 'bannister', 'breakwater', 'dam', 'chainlink_fence', 'picket_fence', 'worm_fence', 'stone_wall', 'grille', 'sliding_door', 'turnstile', 'mountain_tent', 'scoreboard', 'honeycomb', 'plate_rack', 'pedestal', 'beacon', 'mashed_potato', 'bell_pepper', 'head_cabbage', 'broccoli', 'cauliflower', 'zucchini', 'spaghetti_squash', 'acorn_squash', 'butternut_squash', 'cucumber', 'artichoke', 'cardoon', 'mushroom', 'shower_curtain', 'jean', 'carton', 'handkerchief', 'sandal', 'ashcan', 'safe', 'plate', 'necklace', 'croquet_ball', 'fur_coat', 'thimble', 'pajama', 'running_shoe', 'cocktail_shaker', 'chest', 'manhole_cover', 'modem', 'tub', 'tray', 'balance_beam', 'bagel', 'prayer_rug', 'kimono', 'hot_pot', 'whiskey_jug', 'knee_pad', 'book_jacket', 'spindle', 'ski_mask', 'beer_bottle', 'crash_helmet', 'bottlecap', 'tile_roof', 'mask', 'maillot', 'Petri_dish', 'football_helmet', 'bathing_cap', 'teddy', 'holster', 'pop_bottle', 'photocopier', 'vestment', 'crossword_puzzle', 'golf_ball', 'trifle', 'suit', 'water_tower', 'feather_boa', 'cloak', 'red_wine', 'drumstick', 'shield', 'Christmas_stocking', 'hoopskirt', 'menu', 'stage', 'bonnet', 'meat_loaf', 'baseball', 'face_powder', 'scabbard', 'sunscreen', 'beer_glass', 'hen-of-the-woods', 'guacamole', 'lampshade', 'wool', 'hay', 'bow_tie', 'mailbag', 'water_jug', 'bucket', 'dishrag', 'soup_bowl', 'eggnog', 'mortar', 'trench_coat', 'paddle', 'chain', 'swab', 'mixing_bowl', 'potpie', 'wine_bottle', 'shoji', 'bulletproof_vest', 'drilling_platform', 'binder', 'cardigan', 'sweatshirt', 'pot', 'birdhouse', 'hamper', 'ping-pong_ball', 'pencil_box', 'pay-phone', 'consomme', 'apron', 'punching_bag', 'backpack', 'groom', 'bearskin', 'pencil_sharpener', 'broom', 'mosquito_net', 'abaya', 'mortarboard', 'poncho', 'crutch', 'Polaroid_camera', 'space_bar', 'cup', 'racket', 'traffic_light', 'quill', 'radio', 'dough', 'cuirass', 'military_uniform', 'lipstick', 'shower_cap', 'monitor', 'oscilloscope', 'mitten', 'brassiere', 'French_loaf', 'vase', 'milk_can', 'rugby_ball', 'paper_towel', 'earthstar', 'envelope', 'miniskirt', 'cowboy_hat', 'trolleybus', 'perfume', 'bathtub', 'hotdog', 'coral_fungus', 'bullet_train', 'pillow', 'toilet_tissue', 'cassette', "carpenter's_kit", 'ladle', 'stinkhorn', 'lotion', 'hair_spray', 'academic_gown', 'dome', 'crate', 'wig', 'burrito', 'pill_bottle', 'chain_mail', 'theater_curtain', 'window_shade', 'barrel', 'washbasin', 'ballpoint', 'basketball', 'bath_towel', 'cowboy_boot', 'gown', 'window_screen', 'agaric', 'cellular_telephone', 'nipple', 'barbell', 'mailbox', 'lab_coat', 'fire_screen', 'minibus', 'packet', 'maze', 'pole', 'horizontal_bar', 'sombrero', 'pickelhaube', 'rain_barrel', 'wallet', 'cassette_player', 'comic_book', 'piggy_bank', 'street_sign', 'bell_cote', 'fountain_pen', 'Windsor_tie', 'volleyball', 'overskirt', 'sarong', 'purse', 'bolo_tie', 'bib', 'parachute', 'sleeping_bag', 'television', 'swimming_trunks', 'measuring_cup', 'espresso', 'pizza', 'breastplate', 'shopping_basket', 'wooden_spoon', 'saltshaker', 'chocolate_sauce', 'ballplayer', 'goblet', 'gyromitra', 'stretcher', 'water_bottle', 'dial_telephone', 'soap_dispenser', 'jersey', 'school_bus', 'jigsaw_puzzle', 'plastic_bag', 'reflex_camera', 'diaper', 'Band_Aid', 'ice_lolly', 'velvet', 'tennis_ball', 'gasmask', 'doormat', 'Loafer', 'ice_cream', 'pretzel', 'quilt', 'maillot', 'tape_player', 'clog', 'iPod', 'bolete', 'scuba_diver', 'pitcher', 'matchstick', 'bikini', 'sock', 'CD_player', 'lens_cap', 'thatch', 'vault', 'beaker', 'bubble', 'cheeseburger', 'parallel_bars', 'flagpole', 'coffee_mug', 'rubber_eraser', 'stole', 'carbonara', 'dumbbell']
        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()
            # resize label range from [1,1000] to [0,1000),
            # This is required by CrossEntropyLoss
            self.train_labels[:] = [x - 1 for x in self.train_labels]

            self.train_data = np.concatenate(self.train_data)
            [picnum, pixel] = self.train_data.shape
            pixel = int(np.sqrt(pixel / 3))
            self.train_data = self.train_data.reshape((picnum, 3, pixel, pixel))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            [picnum,pixel]= self.test_data.shape
            pixel = int(np.sqrt(pixel/3))

            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()

            # resize label range from [1,1000] to [0,1000),
            # This is required by CrossEntropyLoss
            self.test_labels[:] = [x - 1 for x in self.test_labels]
            self.test_data = self.test_data.reshape((picnum, 3, pixel, pixel))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

