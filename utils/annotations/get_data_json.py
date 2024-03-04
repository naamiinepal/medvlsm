import argparse
import ast
import json
import os
import random

import pandas as pd
from features_from_img import mask_to_overall_bbox


def get_json_data(
    root_dir,
    dataset,
    img_dir_name,
    mask_dir_name,
    op_dir,
    val_per,
    test_per,
    prompt_csv=None,
    default_prompt="",
):
    """Genrates Annotations of segmentation datset in the format requitred from training pytorch.CRIS
    Assumes that we have a root_dir, inside which there are images and masks directories and the images are named sequentially from 0 inside these dirctories,
    Also, this is for the assumption that each image has a single annotation decribing all the objets.

    Args:
        root_dir (string): path of the root dirctory, relative to which directories of images and masks are located
        mask_dir_name (string): name or path or dircetory with segenattion masks, rekative to the root_diir
        op_dir (string): path of the json file of annotations to be output
        val_per(float): valid split percentage
        test_per(float): test split percentage

    Returns:
        string: success message
    """
    # random.seed(444)
    total_data = os.listdir(os.path.join(root_dir, mask_dir_name))
    # total_data = [x for x in total_data if int(x[:-4]) < 900]

    # val_data = random.sample(total_data, int(float(val_per) * len(total_data)))
    # rem_data = [x for x in total_data if x not in val_data]
    # test_data = random.sample(rem_data, int(float(test_per) * len(total_data)))
    # train_data = [x for x in rem_data if x not in test_data]

    val_data = ['208.png', '351.png', '893.png', '504.png', '272.png', '539.png', '174.png', '821.png', '666.png', '130.png', '971.png', '62.png', '390.png', '844.png', '343.png', '223.png', '500.png', '663.png', '192.png', '268.png', '413.png', '580.png', '369.png', '589.png', '114.png', '620.png', '186.png', '509.png', '759.png', '743.png', '447.png', '501.png', '259.png', '476.png', '560.png', '495.png', '962.png', '16.png', '841.png', '480.png', '918.png', '201.png', '203.png', '740.png', '859.png', '764.png', '344.png', '891.png', '716.png', '37.png', '57.png', '100.png', '774.png', '710.png', '119.png', '94.png', '122.png', '538.png', '596.png', '278.png', '619.png', '792.png', '225.png', '482.png', '312.png', '776.png', '139.png', '858.png', '53.png', '787.png', '159.png', '54.png', '611.png', '455.png', '533.png', '448.png', '167.png', '72.png', '852.png', '473.png', '462.png', '356.png', '417.png', '242.png', '652.png', '709.png', '753.png', '779.png', '396.png', '923.png', '392.png', '44.png', '96.png', '568.png', '953.png', '762.png', '95.png', '115.png', '748.png', '530.png'] 
    test_data = ['209.png', '581.png', '75.png', '763.png', '973.png', '820.png', '477.png', '605.png', '69.png', '884.png', '812.png', '84.png', '460.png', '623.png', '12.png', '995.png', '301.png', '324.png', '863.png', '722.png', '836.png', '888.png', '270.png', '726.png', '65.png', '749.png', '409.png', '282.png', '503.png', '948.png', '677.png', '897.png', '761.png', '261.png', '614.png', '502.png', '516.png', '483.png', '542.png', '78.png', '184.png', '739.png', '961.png', '45.png', '207.png', '132.png', '241.png', '239.png', '83.png', '595.png', '89.png', '793.png', '786.png', '747.png', '955.png', '294.png', '850.png', '234.png', '434.png', '309.png', '921.png', '778.png', '700.png', '803.png', '672.png', '497.png', '714.png', '20.png', '522.png', '978.png', '307.png', '529.png', '149.png', '55.png', '843.png', '410.png', '486.png', '540.png', '404.png', '806.png', '916.png', '107.png', '250.png', '899.png', '682.png', '936.png', '305.png', '543.png', '108.png', '262.png', '302.png', '454.png', '718.png', '628.png', '204.png', '901.png', '370.png', '758.png', '675.png', '930.png']
    train_data = ['194.png', '73.png', '244.png', '724.png', '106.png', '909.png', '228.png', '966.png', '189.png', '683.png', '469.png', '894.png', '101.png', '490.png', '18.png', '163.png', '804.png', '355.png', '511.png', '741.png', '375.png', '853.png', '934.png', '996.png', '326.png', '597.png', '660.png', '551.png', '573.png', '946.png', '58.png', '818.png', '654.png', '187.png', '169.png', '133.png', '993.png', '645.png', '662.png', '887.png', '99.png', '7.png', '976.png', '162.png', '195.png', '736.png', '637.png', '3.png', '744.png', '487.png', '196.png', '586.png', '765.png', '907.png', '193.png', '525.png', '348.png', '19.png', '687.png', '517.png', '882.png', '248.png', '817.png', '6.png', '67.png', '790.png', '684.png', '377.png', '382.png', '864.png', '701.png', '895.png', '47.png', '263.png', '197.png', '156.png', '442.png', '470.png', '171.png', '965.png', '445.png', '52.png', '126.png', '438.png', '32.png', '70.png', '650.png', '601.png', '673.png', '521.png', '754.png', '280.png', '453.png', '647.png', '254.png', '711.png', '491.png', '331.png', '702.png', '489.png', '452.png', '855.png', '646.png', '150.png', '861.png', '699.png', '9.png', '519.png', '706.png', '989.png', '904.png', '291.png', '350.png', '910.png', '668.png', '385.png', '2.png', '451.png', '154.png', '616.png', '157.png', '399.png', '419.png', '638.png', '558.png', '364.png', '577.png', '183.png', '506.png', '837.png', '584.png', '25.png', '755.png', '984.png', '667.png', '379.png', '90.png', '251.png', '267.png', '814.png', '206.png', '641.png', '131.png', '617.png', '715.png', '160.png', '255.png', '71.png', '164.png', '49.png', '17.png', '287.png', '940.png', '988.png', '862.png', '237.png', '381.png', '649.png', '877.png', '178.png', '465.png', '39.png', '376.png', '141.png', '548.png', '771.png', '767.png', '523.png', '941.png', '328.png', '719.png', '626.png', '443.png', '340.png', '64.png', '176.png', '389.png', '400.png', '34.png', '110.png', '428.png', '658.png', '703.png', '526.png', '173.png', '585.png', '592.png', '346.png', '534.png', '725.png', '598.png', '933.png', '813.png', '827.png', '200.png', '746.png', '42.png', '137.png', '832.png', '689.png', '138.png', '994.png', '458.png', '210.png', '554.png', '880.png', '750.png', '998.png', '562.png', '79.png', '224.png', '514.png', '252.png', '657.png', '316.png', '950.png', '48.png', '520.png', '216.png', '768.png', '561.png', '840.png', '488.png', '8.png', '214.png', '621.png', '900.png', '479.png', '367.png', '247.png', '732.png', '318.png', '258.png', '643.png', '198.png', '819.png', '905.png', '947.png', '418.png', '265.png', '826.png', '180.png', '775.png', '931.png', '783.png', '277.png', '587.png', '822.png', '352.png', '756.png', '627.png', '766.png', '566.png', '815.png', '723.png', '394.png', '439.png', '190.png', '780.png', '952.png', '926.png', '969.png', '974.png', '135.png', '426.png', '922.png', '185.png', '805.png', '851.png', '590.png', '631.png', '264.png', '871.png', '986.png', '347.png', '240.png', '433.png', '769.png', '127.png', '588.png', '281.png', '655.png', '436.png', '179.png', '873.png', '129.png', '848.png', '760.png', '830.png', '46.png', '527.png', '959.png', '249.png', '275.png', '518.png', '508.png', '59.png', '334.png', '613.png', '670.png', '432.png', '492.png', '957.png', '944.png', '913.png', '698.png', '866.png', '705.png', '11.png', '797.png', '809.png', '898.png', '457.png', '604.png', '985.png', '849.png', '639.png', '505.png', '630.png', '371.png', '583.png', '274.png', '906.png', '729.png', '175.png', '865.png', '235.png', '925.png', '794.png', '903.png', '798.png', '828.png', '839.png', '359.png', '556.png', '968.png', '834.png', '802.png', '113.png', '303.png', '306.png', '353.png', '168.png', '578.png', '329.png', '633.png', '842.png', '745.png', '664.png', '575.png', '742.png', '532.png', '552.png', '594.png', '14.png', '537.png', '161.png', '734.png', '386.png', '88.png', '401.png', '13.png', '733.png', '541.png', '707.png', '213.png', '846.png', '397.png', '120.png', '374.png', '693.png', '437.png', '22.png', '226.png', '143.png', '970.png', '911.png', '337.png', '807.png', '929.png', '295.png', '570.png', '56.png', '555.png', '219.png', '890.png', '727.png', '653.png', '218.png', '338.png', '36.png', '735.png', '467.png', '76.png', '608.png', '142.png', '721.png', '690.png', '345.png', '544.png', '33.png', '738.png', '361.png', '427.png', '170.png', '215.png', '777.png', '825.png', '66.png', '285.png', '569.png', '177.png', '117.png', '144.png', '102.png', '879.png', '717.png', '565.png', '712.png', '874.png', '845.png', '987.png', '293.png', '288.png', '121.png', '981.png', '30.png', '997.png', '975.png', '146.png', '600.png', '373.png', '889.png', '422.png', '405.png', '103.png', '796.png', '498.png', '468.png', '1.png', '549.png', '152.png', '125.png', '314.png', '833.png', '10.png', '935.png', '35.png', '116.png', '421.png', '847.png', '785.png', '81.png', '172.png', '983.png', '731.png', '546.png', '402.png', '870.png', '992.png', '420.png', '349.png', '464.png', '868.png', '140.png', '964.png', '158.png', '896.png', '128.png', '304.png', '220.png', '535.png', '991.png', '576.png', '686.png', '43.png', '854.png', '661.png', '824.png', '211.png', '481.png', '212.png', '91.png', '463.png', '271.png', '362.png', '368.png', '730.png', '191.png', '550.png', '93.png', '335.png', '243.png', '257.png', '423.png', '869.png', '691.png', '789.png', '671.png', '395.png', '648.png', '313.png', '310.png', '980.png', '276.png', '697.png', '914.png', '230.png', '398.png', '867.png', '233.png', '885.png', '105.png', '86.png', '572.png', '564.png', '227.png', '720.png', '383.png', '943.png', '221.png', '231.png', '61.png', '972.png', '928.png', '681.png', '429.png', '547.png', '380.png', '166.png', '624.png', '579.png', '388.png', '232.png', '51.png', '446.png', '23.png', '41.png', '800.png', '696.png', '416.png', '784.png', '531.png', '188.png', '165.png', '466.png', '510.png', '26.png', '881.png', '674.png', '963.png', '801.png', '782.png', '440.png', '496.png', '205.png', '290.png', '461.png', '679.png', '118.png', '924.png', '181.png', '920.png', '829.png', '692.png', '384.png', '283.png', '123.png', '917.png', '878.png', '393.png', '315.png', '378.png', '737.png', '85.png', '810.png', '408.png', '485.png', '536.png', '475.png', '366.png', '155.png', '563.png', '245.png', '609.png', '424.png', '111.png', '406.png', '629.png', '574.png', '136.png', '28.png', '751.png', '431.png', '269.png', '816.png', '875.png', '757.png', '360.png', '823.png', '435.png', '949.png', '321.png', '124.png', '297.png', '459.png', '0.png', '559.png', '238.png', '860.png', '319.png', '644.png', '300.png', '954.png', '908.png', '919.png', '642.png', '289.png', '694.png', '365.png', '256.png', '545.png', '835.png', '695.png', '31.png', '612.png', '912.png', '15.png', '450.png', '308.png', '665.png', '50.png', '781.png', '528.png', '607.png', '704.png', '104.png', '357.png', '80.png', '634.png', '685.png', '622.png', '512.png', '298.png', '831.png', '292.png', '886.png', '40.png', '311.png', '325.png', '415.png', '236.png', '791.png', '553.png', '425.png', '752.png', '217.png', '876.png', '151.png', '591.png', '999.png', '915.png', '222.png', '229.png', '5.png', '474.png', '892.png', '286.png', '772.png', '610.png', '659.png', '253.png', '688.png', '74.png', '979.png', '82.png', '4.png', '977.png', '557.png', '60.png', '680.png', '92.png', '98.png', '449.png', '956.png', '499.png', '571.png', '636.png', '444.png', '515.png', '967.png', '342.png', '982.png', '632.png', '358.png', '202.png', '942.png', '112.png', '656.png', '147.png', '68.png', '441.png', '669.png', '97.png', '606.png', '676.png', '478.png', '77.png', '856.png', '182.png', '109.png', '246.png', '403.png', '625.png', '640.png', '21.png', '296.png', '945.png', '145.png', '412.png', '883.png', '951.png', '339.png', '273.png', '391.png', '958.png', '593.png', '615.png', '333.png', '27.png', '29.png', '838.png', '87.png', '63.png', '153.png', '299.png', '811.png', '927.png', '799.png', '284.png', '524.png', '582.png', '513.png', '24.png', '567.png', '372.png', '323.png', '471.png', '484.png', '602.png', '266.png', '330.png', '317.png', '279.png', '773.png', '327.png', '411.png', '354.png', '938.png', '937.png', '708.png', '939.png', '493.png', '872.png', '990.png', '134.png', '857.png', '678.png', '932.png', '341.png', '494.png', '808.png', '472.png', '320.png', '332.png', '713.png', '635.png', '414.png', '618.png', '407.png', '38.png', '387.png', '199.png', '507.png', '788.png', '322.png', '456.png', '902.png', '960.png', '363.png', '336.png', '430.png', '260.png', '599.png', '651.png', '728.png', '148.png', '795.png', '770.png', '603.png']

    train_json = []
    val_json = []
    test_json = []

    img_ext = os.listdir(os.path.join(root_dir, img_dir_name))[0].split(".")[-1]

    # for dataset in ['Kvaisir-SEG', 'clinicdb-polyp', 'bkai-polyp', 'cvc-300-polyp', 'cvc-colondb-polyp', 'etis-polyp']:
    with open(
        f"/prompts/{dataset}.csv"
    ) as prompt_csv:

        prompt_df = pd.read_csv(prompt_csv)

        prompt_df["p0"] = prompt_df["p0"].fillna("")
        prompt_df["p9"] = prompt_df["p9"].apply(
            lambda x: [n.strip() for n in ast.literal_eval(x)]
        )

        for i, mask in enumerate(total_data):
            mask_path = os.path.join(root_dir, mask_dir_name, mask)
            bbox = mask_to_overall_bbox(mask_path)
            prompt = default_prompt
            seg_id = int(mask_path.split("/")[-1].split(".")[0])

            img_name = mask.split(".")[0] + "." + img_ext
            sent = [{"idx": 0, "sent_id": i, "sent": prompt}]

            mask_df = prompt_df[prompt_df["masks"] == mask]
            mask_df = mask_df.drop(columns=["image", "masks"])
            print(mask_df)
            prompts = mask_df.iloc[0].to_dict()
            op = {
                "bbox": bbox,
                "cat": 0,
                "segment_id": seg_id,
                "img_name": img_name,
                "mask_name": mask,
                "sentences": sent,
                "prompts": prompts,
                "sentences_num": 1,
            }

            if mask in train_data:
                train_json.append(op)
            elif mask in val_data:
                val_json.append(op)
            elif mask in test_data:
                test_json.append(op)

        print(len(train_json), len(val_json), len(test_json))

        os.makedirs(op_dir, exist_ok=True)

        with open(os.path.join(op_dir, "train.json"), "w") as of:
            of.write(json.dumps(train_json))
        with open(os.path.join(op_dir, "val.json"), "w") as of:
            of.write(json.dumps(val_json))
        with open(os.path.join(op_dir, "test.json"), "w") as of:
            of.write(json.dumps(test_json))
        print(f"Json making completed for {dataset}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_data_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--output_data_dir", type=str, required=True)
    parser.add_argument("--valid_per", type=str, required=True)
    parser.add_argument("--test_per", type=str, required=True)
    parser.add_argument("--default_prompt", type=str, required=False)
    args = parser.parse_args()

    print(args)
    get_json_data(
        args.root_data_dir,
        args.dataset_name,
        args.image_dir,
        args.mask_dir,
        args.output_data_dir,
        args.valid_per,
        args.test_per,
        args.default_prompt,
    )

    print("Annotations Created")
