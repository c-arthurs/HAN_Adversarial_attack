###
### Nail model API
### Han Seung Seog (whria78@gmail.com / http://whria.net) / http://medicalphoto.org
### 2017-3-5
### Adapted by Callum Arthurs

import os
import sys
# import matplotlib.pyplot as plt
import caffe
from caffe.proto import caffe_pb2
import numpy as np
import cv2

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

# optimal threshold
# Asan
threshold = [726, 39, 172, 429, 166, 9, 227, 18, 14, 30, 1107, 305]
# Edinburgh
# threshold=[32,996,96,76,7,272,332,3,63,238,10000,10000]

# define alias here
list_alias = []

list_dx = ['abnom', 'abscess', 'acanthosisnigricans', 'acne', 'acneiformeruption', 'acnescar',
           'acrallentiginousnevus',
           'actiniccheilitis', 'actinickeratosis', 'acutegeneralizedexanthematouspustulosis', 'acutegvhd',
           'adultonsetstillsdisease', 'allergiccontactdermatitis', 'allergicvasculitis', 'alopecia',
           'alopeciaareata',
           'amyloidosis', 'androgenicalopecia', 'angioedema', 'angiofibroma', 'angiokeratoma', 'angiolipoma',
           'ashydermatitis', 'ashydermatosis', 'atopicdermatitis', 'atypicalmycobacterialinfection',
           'basalcellcarcinoma', 'basalcellcarcinoma_postop', 'beckernevus', 'behcetdisease', 'bluenevus',
           'bowendisease', 'bowenoidpapulosis', 'bullousdisease', 'bullousdrugeruption', 'bullouspemphigoid',
           'burn',
           'burnscar', 'cafeaulaitmacule', 'calcinosiscutis', 'callus', 'cellulitis',
           'cetuximabinducedacneiformeruption', 'cheilitis', 'chickenpox', 'cholinergicurticaria', 'chroniceczema',
           'chronicgvhd', 'chronicurticaria', 'coldinducedurticaria', 'condyloma',
           'confluentreticulatedpapillomatosis',
           'congenitalnevus', 'connectivetissuedisease', 'contactcheilitis', 'contactdermatitis', 'cutaneoushorn',
           'cyst', 'darkcircle', 'depressedscar', 'dermatitisherpetiformis', 'dermatofibroma', 'dermatomyositis',
           'dilatedpore', 'dirtyneck', 'dohimelanosis', 'drugeruption', 'dyshidroticeczema', 'dysplasticnevus',
           'eczema', 'eczemaherpeticum', 'epidermalcyst', 'epidermalnevus', 'eruptivesyringoma', 'erythemaabigne',
           'erythemaannularecentrifugum', 'erythemamultiforme', 'erythemanodosum', 'exfoliativedermatitis',
           'extramammarypagetdisease', 'fibroma', 'fixeddrugeruption', 'folliculitis', 'fordycespot',
           'foreignbodygranuloma', 'foreignbodyreaction', 'freckle', 'fungalinfection', 'furuncle', 'glomustumor',
           'graftversushostdisease', 'granuloma', 'granulomaannulare', 'guttatepsoriasis', 'handeczema',
           'hemangioma',
           'hematoma', 'henochschonleinpurpura', 'herpessimplex', 'herpeszoster', 'hyperpigmentation',
           'hypersensitivityvasculitis', 'hypertrophicscar', 'hypopigmentation', 'idiopathicguttatehypomelanosis',
           'idreaction', 'impetigo', 'inflammedcyst', 'ingrowingnail', 'insectbite', 'intradermalnevus',
           'irritantcontactdermatitis', 'irritatedlentigo', 'irritatedseborrheickeratosis',
           'juvenilexanthogranuloma',
           'kaposisarcoma', 'keloid', 'keratoacanthoma', 'keratoderma', 'keratosispilaris',
           'langerhanscellhistiocytosis', 'lasertoning', 'lentigo', 'leukemiacutis', 'leukocytoclasticvasculitis',
           'lichenamyloidosis', 'lichennitidus', 'lichenoiddrugeruption', 'lichenplanus', 'lichensimplexchronicus',
           'lichenstriatus', 'lipoma', 'lipomatosis', 'livedoidvasculitis', 'livedoreticularis', 'lmdf',
           'lupuserythematosus', 'lymphangioma', 'lymphoma', 'lymphomatoidpapulosis', 'malignantmelanoma',
           'mastocytoma', 'mastocytosis', 'melanocyticnevus', 'melanonychia', 'melasma', 'metastasis', 'milia',
           'milium', 'molluscumcontagiosum', 'morphea', 'mucocele', 'mucosalmelanoticmacule', 'mucouscyst',
           'mycosisfungoides', 'naildystrophy', 'neurofibroma', 'neurofibromatosis', 'nevus_postop',
           'nevusdepigmentosus', 'nevussebaceus', 'nevusspilus', 'nippleeczema', 'normalnail', 'ntminfection',
           'nummulareczema', 'onycholysis', 'onychomycosis', 'organoidnevus', 'otanevus', 'otherdermatitis',
           'pagetsdisease', 'palmoplantarpustulosis', 'panniculitis', 'papularurticaria', 'parapsoriasis',
           'paronychia',
           'pemphigusfoliaceus', 'pemphigusvulgaris', 'perioraldermatitis', 'photosensitivedermatitis',
           'pigmentedcontactdermatitis', 'pigmentednevus', 'pigmentedprogressivepurpuricdermatosis', 'pilarcyst',
           'pilomatricoma', 'pityriasisalba', 'pityriasislichenoideschronica',
           'pityriasislichenoidesetvarioliformisacuta', 'pityriasisrosea', 'pityriasisrubrapilaris', 'poikiloderma',
           'pompholyx', 'porokeratosis', 'poroma', 'portwinestain', 'postinflammatoryhyperpigmentation',
           'prurigonodularis', 'prurigopigmentosa', 'pruritus', 'pseudolymphoma', 'psoriasis', 'puppp', 'purpura',
           'pustularpsoriasis', 'pyodermagangrenosum', 'pyogenicgranuloma', 'rhielmelanosis', 'rosacea',
           'rupturedcyst',
           'sarcoidosis', 'scabies', 'scar', 'scar_postlaser', 'scar_postop', 'scc_postop', 'scleroderma',
           'sebaceoushyperplasia', 'seborrheicdermatitis', 'seborrheickeratosis', 'skintag', 'softfibroma',
           'squamouscellcarcinoma', 'staphylococcalscaldedskinsyndrome', 'stasisdermatitis',
           'steatocystomamultiplex',
           'steroidrosacea', 'striaedistensae', 'subcutaneousnodule', 'subungalhematoma', 'sweetsyndrome',
           'syringoma',
           'systemiccontactdermatitis', 'systemiclupuserythematosus', 'tattoo', 'telangiectasia', 'tineacorporis',
           'tineafaciale', 'tineapedis', 'toxicepidermalnecrolysis', 'traumaticfatnecrosis', 'traumatictattoo',
           'ulcer',
           'urticaria', 'urticarialvasculitis', 'urticariapigmentosa', 'varicella', 'vascularmalformation',
           'vasculitis', 'venouslake', 'venousmalformation', 'verrucaplana', 'viralexanthem', 'vitiligo', 'wart',
           'wrinkle', 'xanthelasma', 'xanthogranuloma', 'xanthoma', 'xeroticeczema']

main_dx2 = ['Malignant melanoma', 'Basal cell carcinoma', 'Squamous cell carcinoma', 'Intraepithelial carcinoma',
            'Pyogenic granuloma', 'Seborrheic keratosis', 'Melanocytic nevus', 'Actinic keratosis',
            'Dermatofibroma',
            'Hemangioma', 'Wart', 'Lentigo']
main_dx = ['malignantmelanoma', 'basalcellcarcinoma', 'squamouscellcarcinoma', 'bowendisease', 'pyogenicgranuloma',
           'seborrheickeratosis', 'pigmentednevus', 'actinickeratosis', 'dermatofibroma', 'hemangioma', 'wart',
           'lentigo']


def senderror(msg):
    print(msg)
    sys.exit(0)


def getname(i):
    for j, dx_ in enumerate(main_dx):
        if (dx_ == list_dx[i]): return main_dx2[j]
    return ""


def get_basenames(img_path):
    basenames = []
    dirname = os.path.dirname(img_path)
    for alias_ in list_alias:
        dirname = dirname.replace(alias_[0], alias_[1])
    olddir = ''
    while dirname != '' and dirname != '/' and olddir != dirname:
        if ('lesion_' not in os.path.basename(dirname)):
            basenames += [os.path.basename(dirname)]
        olddir = dirname
        dirname = os.path.dirname(dirname)
    return basenames


def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    # Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    # Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

    return img


def loadcaffemodel(modelbasepath, modelname, deployname, test_img_paths):
    mean_blob = caffe_pb2.BlobProto()
    with open(os.path.join(modelbasepath, 'mean224x224.binaryproto'), 'rb') as f:
        mean_blob.ParseFromString(f.read())
    mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
        (mean_blob.channels, mean_blob.height, mean_blob.width))

    # Read model architecture and trained model's weights
    print(os.path.join(modelbasepath, deployname), os.path.join(modelbasepath, modelname + '.caffemodel'))
    net = caffe.Net(os.path.join(modelbasepath, deployname), os.path.join(modelbasepath, modelname + '.caffemodel'),
                    caffe.TEST)

    # Define image transformers
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', mean_array)
    transformer.set_transpose('data', (2, 0, 1))

    result = []
    for img_path in test_img_paths:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)

        net.blobs['data'].data[...] = transformer.preprocess('data', img)
        out = net.forward()
        pred_probas = out['prob']
        result += [(img_path, pred_probas[0].tolist())]
        print("predicting  - ", img_path)
        # print (img_path,pred_probas[0])
    return result


def loadmodel(train_dataset, train_type, exp_num, test_img_paths):
    print("train_dataset - ", train_dataset)
    if (train_type == 0):  # ResNet-152 alone
        deployname = 'deploy.prototxt'
        print("modelpath = ", os.path.join(os.getcwd(), 'model', train_dataset))
        model_path = os.path.join(os.getcwd(), 'model', train_dataset)
        name_caffemodel = ''
        start_iter = 0
        if (train_dataset == 'asan'): start_iter = 59024
        if (train_dataset == 'asanplus'): start_iter = 70615
        name_caffemodel += str((start_iter + exp_num))
        return loadcaffemodel(model_path, name_caffemodel, deployname, test_img_paths)


def get_max_diagnosis(diagnosis):
    """
    get the max diagnosis value from a list of names
    :param diagnosis: list [fname, [disease, number], [disease, number],...]
    :return: the max diagnosis and name - ['Wart', 0.9997361302375793]
    """
    ints = [d[1] for d in diagnosis[1:]]
    idx = ints.index(max(ints))  # get index of max value
    final = diagnosis[idx + 1]  # added the plus one to account for name at idx 0
    final.insert(0, os.path.split(diagnosis[0])[1])
    return final


def run():
    gpu_device = -1  # default = CPU
    if (len(sys.argv) > 5): gpu_device = int(sys.argv[5])
    if (gpu_device == -1):
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(gpu_device)
    print("GPU - ", gpu_device)

    # test_path   path1;path2;path3
    if (len(sys.argv) > 1): test_path_list = str(sys.argv[1]).split(';')
    # train_dataset asan90;asan10
    if (len(sys.argv) > 2): train_dataset = str(sys.argv[2])
    # train_type 0:resnet
    train_type = 0
    if (len(sys.argv) > 3): train_type = int(sys.argv[3])
    exp_num = 0
    if (len(sys.argv) > 4): exp_num = int(sys.argv[4])

    # list_dx

    for test_path in test_path_list:
        if (os.path.exists(test_path) == False):
            print(test_path + ' is not exist')
            sys.exit(0)
        print("Test Path : ", test_path)

    ### Get list of indexes ###
    print("DX List : ", list_dx)

    test_img_paths = []
    for test_path in test_path_list:
        for root, dirs, files in os.walk(test_path):
            for fname in files:
                ext = (os.path.splitext(fname)[-1]).lower()
                if ext == ".jpg" or ext == ".jpeg" or ext == ".gif" or ext == ".png": test_img_paths += [
                    os.path.join(root, fname)]

        if (len(test_img_paths) == 0):
            print("No image (.jpg .gif .png) exist at " + test_path)
            sys.exit(0)

        # run model
    modelnail = loadmodel(train_dataset, train_type, exp_num, test_img_paths)

    final_result = []
    for i, img_path in enumerate(test_img_paths):
        model_result = []
        temp = []

        print("hi - ", img_path)
        for modelnail_ in modelnail:
            if (modelnail_[0] == img_path):
                model_result = modelnail_[1]
                # print("result - ", model_result)

        # get right index from folder name
        right_dx_index = -1
        right_dx_name = ''
        for i, dx_ in enumerate(main_dx2):
            if dx_ in get_basenames(img_path):
                right_dx_name = main_dx[i]
                for j, dx2_ in enumerate(list_dx):
                    if (dx2_ == right_dx_name):
                        right_dx_index = j
        # print "Diagnosis identified by the name of folder: ",list_dx[right_dx_index]

        final_result += [(img_path, model_result, right_dx_index)]
    #
    # Print Result
    #

    results = []

    countall = 0.0
    correct = 0.0

    all_results = []

    for final_ in final_result:
        countall += 1
        diagnosis = [final_[0]]
        print("\n")
        print("Image path : %s" % final_[0])
        # print("Correct Diagnosis : ")
        # if (final_[2] == -1):
        #     print("  %s" % "unknown")
        # else:
        #     print("  %s" % getname(final_[2]))
        # print("Model's Prediction : ")
        f_ = []
        for i, p_ in enumerate(final_[1]):
            thres_ = 10000
            for j, dx_ in enumerate(main_dx):
                if (dx_ == list_dx[i]):
                    thres_ = threshold[j]
            if (p_ * 10000 > thres_):
                f_ += [(p_, getname(i))]
                if i == final_[2]: correct += 1
        f_ = sorted(f_, reverse=True)

        for f in f_:
            print("  R/O %s" % f[1])
        print("Model's Output : ")
        for i, p_ in enumerate(final_[1]):
            for j, dx_ in enumerate(main_dx):
                if (dx_ == list_dx[i]):
                    print("  %s : %.4f" % (getname(i), p_))
                    diagnosis.append([getname(i), p_])
                    results.append((getname(i), p_, final_[0]))

        final_diagnosis = get_max_diagnosis(diagnosis)
        all_results.append(final_diagnosis)
        print(final_diagnosis)
        if final_diagnosis[1] == "Malignant melanoma":
            correct += 1


    print("Correct ratio : %.1f (%d / %d)" % (correct / countall * 100, correct, countall))
    return all_results


if __name__ == "__main__":
    run()
