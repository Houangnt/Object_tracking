def split_folders_to_train(data_folder, output_folder):
    import splitfolders

    splitfolders.ratio(
        data_folder, output=output_folder, seed=1337, ratio=(0.8, 0.2)
    )

def copy_files_to_another_folder(original, target):
    import shutil
    shutil.copyfile(original, target)

def remove_file(filepath):
    import os 
    if os.path.isfile(filepath): 
        print(f"{filepath}")
        os.remove(filepath)

def remove_files_list_in_txtfile(path, txt_file):
    import os 
    os.chdir(path)
    flist = open(txt_file)
    for f in flist:
        fname = f.rstrip() 
        if os.path.isfile(fname): 
            print(f"{fname}")
            os.remove(fname)
    flist.close()

def img2label_annotation(img_anno):
    exp = img_anno.split(".")[-1]
    label_anno = img_anno.replace("." + exp, ".txt")
    return label_anno

def label2img_annotation(label_anno):
    exp = label_anno.split(".")[-1]
    img_anno = label_anno.replace("." + exp, ".jpg")
    return img_anno

def check_label_duplicate(label_path, save_path):
    uniqlines = set(open(label_path, 'r').readlines())
    fw = open(save_path, 'w')
    fw.writelines(uniqlines)
    fw.close()

def check_image_valid(folder_path):
    from PIL import ImageFile, Image

    count = 0
    for img_path in paths.list_images(folder_path):
        img = Image.open(img_path)
        img.verify()

        print("Done.")
        count = count + 1
    print("Folder has : {} images", count)

def padding_image(image, padding=30, color=(0, 255, 255)):
    new_img = cv2.copyMakeBorder(
        image,
        top=padding,
        bottom=padding,
        left=padding,
        right=padding,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )
    return new_img

def get_boxes(image, coordinates):
    boxes = []
    for coor in coordinates:
        print('coordinates: ', coor)
        top_left = coor[0]
        bottom_right = coor[2]
        print(f'top left {top_left}  bottom_right {bottom_right}')
        box = image[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0], :]
        boxes.append(box)
    return boxes

def standardlize_value_coordinates(img_siz, coordinates):
    for coor in coordinates:
        if coor[0] < 0 :
            coor[0] = 0
        elif coor[0] > img_siz[1] :
            coor[0] = img_siz[1]
        if coor[1] < 0 :
            coor[1] = 0
        elif coor[1] > img_siz[0] :
            coor[1] = img_siz[0] 

def draw_boxes(image, coordinates):
    """
        Visualize
    """
    import cv2
    color = (255, 0, 0)
    clc_txt = (0, 0, 255)
    thickness = 1
    thickness_txt = 1

    for coor in coordinates:
        print('coordinates: ', coor)
        # x_coors = coor[0]
        # y_coors = coor[1]
        top_left = coor[0]
        bottom_right = coor[2]
        print(f'top left {top_left}  bottom_right {bottom_right}')
        # for i in range(4):
            # point = (int(x_coors[i]), int(y_coors[i]))
            # cv2.circle(image, point, thickness, color, thickness)
            # cv2.putText(image, str(i), point, cv2.FONT_HERSHEY_SIMPLEX, 1, clc_txt, thickness_txt)
        for point in coor:
            cv2.circle(image, (int(point[0]), int(point[1])), thickness, (0, 255, 0), thickness)

        cv2.rectangle(image, (int(top_left[0]), int(top_left[1])), \
                            (int(bottom_right[0]), int(bottom_right[1])), color, thickness)

def draw_labels(image, coordinates, labels):
    """
        Visualize labels
    """
    import cv2
    color = (0, 23, 0)
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5

    for idx, coor in enumerate(coordinates):
        top_left = coor[0]
        bottom_left = coor[3]
        cv2.putText(image, labels[idx], (int(bottom_left[0]), int(bottom_left[1] - 15)), \
                            font, fontScale, color, thickness, cv2.LINE_AA)


def get_filename_from_path(file_path):
    file_name = file_path.split("/")[-1]
    exp = file_name.split(".")[-1]
    return file_name.replace("." + exp, "")

def subtract2images(path1, path2, is_show, is_save, path_save):
    import cv2
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    subtracted = cv2.subtract(img1, img2)
    if is_show: 
        subtracted = cv2.resize(subtracted, (960, 960))
        cv2.imshow('subtracted', subtracted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if is_save:
        name_save = get_filename_from_path(path1) + "_" + get_filename_from_path(path2) + ".jpg"
        import os
        path_save = os.path.join(path_save, name_save)
        cv2.imwrite(path_save, subtracted)

def check_image_duplicate(folder_path):
    from imutils import paths
    img_paths = list(paths.list_images(folder_path))

    for i in range(len(img_paths) - 1):
        for j in range(i + 1, len(img_paths)):
            subtract2images(img_paths[i], img_paths[j], False, True, folder_path)

def video2images(vid_path, folder_path):
    import cv2
    import os
    from math import log10
    vidcap = cv2.VideoCapture(vid_path)
    count = 1
    while(vidcap.isOpened()):
        ret, frame = vidcap.read()
        if ret == False:
            break
        file_name = "0" * int((6 - int(log10(count)) -1)) + "{}.jpg".format(count) 
        cv2.imwrite(os.path.join(folder_path, file_name), frame)    
        # cv2.imshow('frame', frame)
        # print('Read a new frame: ', ret)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        count += 1
    vidcap.release()

def labelme2groundtruth(label_path, save_path):
    import os
    import json
    fw = open(save_path, "w")
    for json_file in os.listdir(label_path):
        if json_file.split('.')[-1] == 'json':
            json_path = os.path.join(label_path, json_file)
            data = json.load(open(json_path))
            label_list = []
            label_line = json_file.replace(".json", ".jpg")
            for i in range(len(data['shapes'])):
                label = data['shapes'][i]['label']
                label_list.append(label)
            label_list = list(set(label_list))
            print('before sieve: ', label_list)
            label_list = substring_sieve(label_list)
            print('after sieve: ', label_list)

            label_list = [label for label in label_list if len(label) > 8]
            
            # if len(label_list):
            for label in label_list:
                label_line = label_line + "\t" + str(label)
            print(label_line)
            fw.writelines(label_line + "\n")
            # else: 
                # remove_file(json_path)
    fw.close()     

def substring_sieve(string_list):
    string_list.sort(key=lambda s: len(s), reverse=True)
    out = []
    for s in string_list:
        if not any([s in o for o in out]):
            out.append(s)
    return out

def copy_jpg_respect_json():
    import glob
    import os
    label_folder = "/home/hanhpm2/Downloads/MobileDatasets/train_gen/new_dataset/train/labels/"
    image_folder = '/home/hanhpm2/Downloads/MobileDatasets/train_gen/label_json/'
    save_folder = "/home/hanhpm2/Downloads/MobileDatasets/train_gen/new_dataset/train/images/"

    json_files =  glob.glob(label_folder + "*.json", recursive = True)
    json_files = [j_file.split("/")[-1] for j_file in json_files]
    jpg_files = [j_file.split("/")[-1] for j_file in json_files]
    print(len(jpg_files))
    print(len(json_files))

    for json in json_files:
        j_name = json.replace(".json", ".jpg")
        # print(j_name)
        copy_files_to_another_folder(os.path.join(image_folder, j_name), os.path.join(save_folder, j_name))
        # copy_files_to_another_folder(os.path.join(root_folder, json), os.path.join(label_folder, json))

def copy_jpg_respect_txt():
    import glob
    import os
    label_folder = "/home/hanhpm2/Downloads/MobileDatasets/train_gen_2/new_dataset/train/labels/"
    image_folder = '/home/hanhpm2/Downloads/MobileDatasets/train_gen_2/label_json/'
    save_folder = "/home/hanhpm2/Downloads/MobileDatasets/train_gen_2/new_dataset/train/images/"
    print(label_folder)
    txt_files =  glob.glob(label_folder + "/*.txt", recursive = True)
    print(txt_files)
    txt_files = [j_file.split("/")[-1] for j_file in txt_files]
    jpg_files = [j_file.split("/")[-1] for j_file in txt_files]
    print(len(jpg_files))
    print(len(txt_files))

    for txt in txt_files:
        j_names = [txt.replace(".txt", ".jpg"), txt.replace(".txt", ".png"), txt.replace(".txt", ".jpeg")]
        j_exists = [os.path.exists(os.path.join(image_folder, j_n)) for j_n in j_names]
        j_name_exists = [ j_n for idx, j_n in enumerate(j_names) if j_exists[idx] ]
        if len(j_name_exists) > 0:
            j_name = j_name_exists[0]
        else:
            continue
        print(j_name)
        copy_files_to_another_folder(os.path.join(image_folder, j_name), os.path.join(save_folder, j_name))

def find_json_with_version(json_folder, save_path1, save_path2, version):
    import os
    import json
    img_folder = json_folder.replace("labels", "images")
    for json_file in os.listdir(json_folder):
        if json_file.split('.')[-1] == 'json':
            print(json_file)
            json_path = os.path.join(json_folder, json_file)
            data = json.load(open(json_path))
            img_name = data['imagePath']
            json_name = data['imagePath'].replace(".jpg", ".json")
            
            print(img_folder)

            if data['version'] == version:
                copy_files_to_another_folder( \
                    os.path.join(json_folder, json_name), \
                    os.path.join(save_path1, json_name))
                copy_files_to_another_folder( \
                    os.path.join(img_folder, img_name), \
                    os.path.join(save_path1, img_name))
            else:
                copy_files_to_another_folder( \
                    os.path.join(json_folder, json_name), \
                    os.path.join(save_path2, json_name))
                copy_files_to_another_folder( \
                    os.path.join(img_folder, img_name), \
                    os.path.join(save_path2, img_name))

def standardlize_json_version(json_folder, save_folder):
    import os
    import json
    for json_file in os.listdir(json_folder):
        if json_file.split('.')[-1] == 'json':
            json_path = os.path.join(json_folder, json_file)
            data = json.load(open(json_path))
            img_name = data['imagePath']
            json_name = data['imagePath'].replace(".jpg", ".json")

            version = "3.16.7"
            line_color = None
            fill_color = None
            shape_type = "rectangle"
            flags = {}
            lineColor = [0, 255, 0, 128]
            fillColor = [255, 0, 0, 128]

            shapes = []
            data_shapes = data['shapes']
            for i in range(len(data_shapes)):
                data_shape = data_shapes[i]
                shape = {
                    "label" : data_shape["label"], 
                    "line_color" : line_color, 
                    "fill_color" : fill_color, 
                    "points" : data_shape["points"], 
                    "shape_type" : shape_type, 
                    "flags" : flags
                }
                shapes.append(shape)

            dictionary = {
                "version" : version, 
                "flags" : flags, 
                "shapes" : shapes,
                "lineColor" : lineColor, 
                "fillColor" : fillColor, 
                "imagePath" : data["imagePath"],
                "imageData" : data["imageData"], 
                "imageHeight": data["imageHeight"], 
                "imageWidth" : data["imageWidth"]
            }

            json_object = json.dumps(dictionary, indent = 4)
            with open(os.path.join(save_folder, json_name), "w") as outfile:
                outfile.write(json_object)


def read_all_lines(f): 
    import os
    if isinstance(f, str) and os.path.exists(f):
        fr = open(f, "r")
    else: 
        fr = f
    lines = []
    while True: 
        line = fr.readline()
        if not line: 
            break 
        lines.append(line) 
    return lines

def compare_two_file_content(file_A, file_B, operators): 
    if isinstance(file_A, str) and isinstance(file_B, str):
        fr_A = open(file_A, "r")
        fr_B = open(file_B, "r")
        set_A = set(read_all_lines(fr_A))
        set_B = set(read_all_lines(fr_B))
    elif (isinstance(file_A, list) and isinstance(file_B, list)) or \
        (isinstance(file_A, set) and isinstance(file_B, set)):
        set_A = set(file_A)
        set_B = set(file_B)
 
    if not isinstance(operators, list): 
        operators = [operators]
    
    results = {}

    for ope in operators: 
        if ope == "union": 
            uni = set_A.union(set_B)
            results.update({"union" : uni})
        elif ope == "intersection": 
            intersect = set_A.intersection(set_B) 
            results.update({"intersection" : intersect})
        elif ope == "difference":
            differ_A_from_B = set_A.difference(set_B)
            differ_B_from_A = set_B.difference(set_A)
            results.update({'difference' : [differ_A_from_B, differ_B_from_A]})
        else: 
            assert "Set operator must be union, intersection or difference!"

    if isinstance(file_A, str) and isinstance(file_B, str):
        fr_A.close()
        fr_B.close()

    return results

def compare_two_model_barcode_using_GT():
    import os 

    result_folder_A = "/home/hanhpm2/Downloads/BarcodeDatasets/Flash-Scanner/raw_dataset/ground_truth/result/020822/"
    result_folder_B = "/home/hanhpm2/Downloads/BarcodeDatasets/Flash-Scanner/raw_dataset/ground_truth/result/020822/pruned"
    save_folder = "/home/hanhpm2/Downloads/BarcodeDatasets/Flash-Scanner/raw_dataset/ground_truth/result/020822/compare/"
    files = ["true_all_case", "miss_case", "redundant_case", "none_case"] 
    ext = ".txt"

    for f_name in files: 
        compared_results = compare_two_file_content(
                            file_A = os.path.join(result_folder_A, f_name + ext), 
                            file_B = os.path.join(result_folder_B, f_name + ext), 
                            operators = ["intersection", "difference"]
                            )
        with open(os.path.join(save_folder, f_name + "_intersect" + ext), "w") as fw: 
            for line in compared_results["intersection"]:
                fw.write(line) 
        with open(os.path.join(save_folder, f_name + "_differ_A_from_B" + ext), "w") as fw: 
            for line in compared_results["differences"][0]:
                fw.write(line) 
        with open(os.path.join(save_folder, f_name + "_differ_B_from_A" + ext), "w") as fw: 
            for line in compared_results["differences"][1]:
                fw.write(line) 

def show_images_from_list(folder_path, img_list):
    import cv2
    import os
    for img_name in img_list:
        img_name = img_name.split("\n")[0] 
        img_path = os.path.join(folder_path, img_name)
        print(img_path)
        print(os.path.exists(img_path))
        img = cv2.imread(img_path)
        img = cv2.resize(img, (1280, 720))
        cv2.imshow(img_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def get_all_filename_in_folder(folder_path, save_path, ext = ".jpg"):
    import os
    from imutils import paths
    fw = open(save_path, "w")
    for f in os.listdir(folder_path): 
        f_name = os.path.basename(f)
        # f_name = f_name.replace(ext, "")
        fw.writelines(f_name)
        fw.writelines("\n")

    fw.close()

def duplicate_file_json():
    image_folder = '/home/hanhpm2/Downloads/CIC/img_cic'
    import os 
    import json 
    import base64

    all_files = os.listdir(image_folder)
    json_files = [f for f in all_files if ".json" in f]
    img_files = [f for f in all_files if ".jpg" in f]

    
    for json_file in json_files: 
        print(json_file)
        for img_file in img_files:
            img_name = img_file.split(".")[0]
            if img_name in json_file:
                continue 
            img_path = os.path.join(image_folder, img_file)
            imageData = base64.b64encode(open(img_path, "rb").read()).decode('utf-8')

            json_path = os.path.join(image_folder, img_name + ".json")
            f = open(json_path, 'r+')
            data = json.load(f)
            data["imageData"] = imageData
            data["imagePath"] = img_file

            f.seek(0)        
            json.dump(data, f, indent=4)
            f.truncate()