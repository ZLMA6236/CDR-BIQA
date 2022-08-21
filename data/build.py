
import random
import data_loader

folder_path = {
    'lsrq': r'G:\DATASETS\LSRQ\\',
    'livec': 'F:\DataSets\ChallengeDB_release\\',
    'bid': 'F:\DataSets\BID\ImageDatabase\\',
    'koniq-10k': 'F:\DataSets\koniq\\',
    'spaq': 'G:\DATASETS\SPAQ zip\SPAQ\\',
    'flive': r'E:\flive\\',
}
img_num = {
    'lsrq': list(range(0, 500000)),
    'livec': list(range(0, 1162)),
    'bid': list(range(0, 586)),
    'koniq-10k': list(range(0, 10073)),
    'spaq': list(range(0, 11125)),
    'flive': list(range(0, 39807)),
}

def build_loader_lsrq(config):
    random.seed(2)
    train_idx = img_num['lsrq']
    test_idx_livec = img_num['livec']
    test_idx_bid = img_num['bid']
    test_idx_koniq = img_num['koniq-10k']
    test_idx_spaq = img_num['spaq']
    test_idx_flive = img_num['flive']

    path = folder_path['lsrq']
    path_livec = folder_path['livec']
    path_bid = folder_path['bid']
    path_koniq = folder_path['koniq-10k']
    path_spaq = folder_path['spaq']
    path_flive = folder_path['flive']

    train_loader = data_loader.DataLoader('lsrq', path, train_idx, config.DATA.IMG_SIZE, config.DATA.TRAIN_PATCH_NUMBER, batch_size=config.DATA.BATCH_SIZE, istrain=True)
    test_loader_livec = data_loader.DataLoader('livec', path_livec, test_idx_livec, config.DATA.IMG_SIZE, config.DATA.TEST_PATCH_NUMBER, istrain=False)
    test_loader_bid = data_loader.DataLoader('bid', path_bid, test_idx_bid, config.DATA.IMG_SIZE, config.DATA.TEST_PATCH_NUMBER, istrain=False)
    test_loader_koniq = data_loader.DataLoader('koniq-10k', path_koniq, test_idx_koniq, config.DATA.IMG_SIZE, config.DATA.TEST_PATCH_NUMBER, istrain=False)
    test_loader_spaq = data_loader.DataLoader('spaq', path_spaq, test_idx_spaq, config.DATA.IMG_SIZE, config.DATA.TEST_PATCH_NUMBER, istrain=False)
    test_loader_flive = data_loader.DataLoader('flive', path_flive, test_idx_flive, config.DATA.IMG_SIZE, config.DATA.TEST_PATCH_NUMBER, istrain=False)

    data_loader_train = train_loader.get_data()
    data_loader_val_livec = test_loader_livec.get_data()
    data_loader_val_bid = test_loader_bid.get_data()
    data_loader_val_koniq = test_loader_koniq.get_data()
    data_loader_val_spaq = test_loader_spaq.get_data()
    data_loader_val_flive = test_loader_flive.get_data()

    return data_loader_train, data_loader_val_livec,data_loader_val_bid,data_loader_val_koniq,\
           data_loader_val_spaq,data_loader_val_flive

def build_loader_single(config):
    # random.seed(2)
    sel_num = img_num[config.DATA.DATASET]
    path = folder_path[config.DATA.DATASET]
    random.shuffle(sel_num)
    train_idx = sel_num[0:int(round(0.8 * len(sel_num)))]
    test_idx = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]
    train_loader = data_loader.DataLoader(config.DATA.DATASET, path, train_idx, config.DATA.IMG_SIZE, config.DATA.TRAIN_PATCH_NUMBER, batch_size=config.DATA.BATCH_SIZE, istrain=True)
    test_loader = data_loader.DataLoader(config.DATA.DATASET, path, test_idx, config.DATA.IMG_SIZE, config.DATA.TEST_PATCH_NUMBER, istrain=False)
    data_loader_train = train_loader.get_data()
    data_loader_val = test_loader.get_data()

    return data_loader_train, data_loader_val