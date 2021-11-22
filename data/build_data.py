import subprocess, shlex
from pathlib import Path
import os
import tqdm
import pickle
from typing import Tuple, List
import numpy as np
import pandas as pd
import traceback

OGVC_TOP_DIR='/autofs/diamond/share/corpus/OGVC/Vol2/Acted/wav'
OPENSMILE_PATH='../tools/opensmile-3.0-linux-x64'
OPENSMILE_EXEC_PATH=os.path.join(OPENSMILE_PATH, 'bin', 'SMILExtract')
OPENSMILE_CONF_PATH=os.path.join(OPENSMILE_PATH, 'config', 'compare16', 'ComParE_2016.conf')
CSV_DIR = './OGVC_LLD_CSV'

PICKLE_DIR = 'OGVC_LLD_PICKLE'
PICKLE_PATH = os.path.join(PICKLE_DIR, 'data.pkl')
PICKLE_NORM_PATH = os.path.join(PICKLE_DIR, 'data_norm.pkl')

INFO_CSV_PATH = 'info.csv'

def get_wav_list(topdir: str=OGVC_TOP_DIR):
    wav_list = []
    for p in Path(topdir).rglob('*.wav'):
        wav_list.append(p.as_posix())
    wav_list = sorted(wav_list)
    return wav_list

def extract_lld(in_wav_file_path, out_csv_file_path, instname = None):
    if instname is None:
        basename = os.path.basename(in_wav_file_path)
        body, _ = os.path.splitext(basename)
        instname = body

    cmd = "{} -C {} -I {} -lldcsvoutput {} -instname {}".format(
        OPENSMILE_EXEC_PATH, OPENSMILE_CONF_PATH, in_wav_file_path, out_csv_file_path, instname
    )
    args = shlex.split(cmd)
    subprocess.run(args, stderr=subprocess.DEVNULL)

def extract_all_lld(in_wav_topdir: str=OGVC_TOP_DIR, csv_dir=CSV_DIR):
    wav_list = get_wav_list(in_wav_topdir)
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir, mode=0o755)
    print("extract LLD features with openSMILE: ")
    for wav_path in tqdm.tqdm(wav_list):
        basename = os.path.basename(wav_path)
        body, _ = os.path.splitext(basename)
        instname = body
        csv_path = os.path.join(csv_dir, body + '.csv') 
        extract_lld(wav_path, csv_path)

def read_csv(csv_file_path: str) -> Tuple[str, str, np.ndarray]:
    """CSVファイルを読み込む

    Args:
        csv_file_path (str): 読み込むCSVファイルのパス

    Returns:
        str: name（元のファイル名のボディ部分）
        np.ndarray: 特徴ベクトルの時系列データ（L x D)
    """
    df = pd.read_csv(csv_file_path, sep=';')

    # name. シングルクォートで囲まれているので取り除く
    name = df['name'][0][1:-1]

    # 特徴量. name, frameTime 列は取り除く
    x = df.iloc[:, 2:].to_numpy()

    return (name, x)

def read_all_csv(top_dir: str=CSV_DIR) -> List[Tuple[str, np.ndarray]]:
    """すべてのCSVファイルを読み込む.

    Args:
        top_dir (str, optional): CSVファイルがあるディレクトリのパス. Defaults to DATA_TOP_DIR.

    Returns:
        List[Tuple[str, np.ndarray]]: read_csv関数の戻り値のリスト
    """
    result = []
    print("read all csv files in {}:".format(top_dir))
    path_list = []
    for path in Path(top_dir).rglob('*.csv'):
        path_list.append(path.as_posix())
    path_list = sorted(path_list)

    for path in tqdm.tqdm(path_list):
        fullpath = path
        label, x = read_csv(fullpath)

        result.append((label, x,))

        # # デバッグのため
        # if len(result) > 10:
        #     break

    return result

def normalize(data: List[Tuple[str, np.ndarray]]) -> List[Tuple[str, np.ndarray]]:
    """read_all_csv関数で読み込んだデータを，全データの平均値，分散で正規化する

    Args:
        data (List[Tuple[str, np.ndarray]]): read_all_csv関数で読み込んだ全データ

    Returns:
        List[Tuple[str, np.ndarray]]: 正規化後のデータ
    """

    count_list = []
    mean_list = []
    std_list = []

    print("data normalization step 1. calculating entire mean and std:")
    for label, x in tqdm.tqdm(data):
        count_list.append(len(x))
        mean_list.append(x.mean(axis=0))
        std_list.append(x.std(axis=0))

    mean = (np.matmul(
           np.array(count_list).reshape(1, -1), np.stack(mean_list)
        ) / np.sum(count_list)).flatten()
    std = np.sqrt(np.matmul(
            np.array(count_list).reshape(1, -1), np.square(np.stack(std_list))
        ) / np.sum(count_list)).flatten()

    print("data noramalization step 2. normalization:")
    result = []
    for label, x in tqdm.tqdm(data):
        x = (x - mean) / std
        result.append((label, x))

    return result

def load_data(filepath=PICKLE_PATH, reload=False):
    """データをロードする．既に読み込み済でpickleファイルが作られていればそれを読み込み返す．

    Args:
        filepath (str, optional): pickleファイルのパス. Defaults to PICKLE_PATH.
        reload (bool, optional): 強制的に再読み込みをするかどうか. Defaults to False.

    Returns:
        read_all_csv関数と同様.
    """

    data = None

    if not reload and os.path.exists(filepath):
        try:
            data = pickle.load(open(filepath, 'rb'))
        except Exception:
            print(traceback.format_exc())

    if not data:
        data = read_all_csv()
        pickle.dump(data, open(filepath, 'wb'))

    return data

def load_norm_data(filepath=PICKLE_NORM_PATH, reload=False, orig_filepath=PICKLE_PATH):
    """データをロードする．既に読み込み済でpickleファイルが作られていればそれを読み込み返す．

    Args:
        filepath (str, optional): pickleファイルのパス. Defaults to PICKLE_PATH.
        reload (bool, optional): 強制的に再読み込みをするかどうか. Defaults to False.
        orig_filepath (str, optional): 内部でload_dataに渡されるpickleファイルのパス． Defaults to PICKLE_PATH.

    Returns:
        read_all_csv関数の戻り値と同様．ただし正規化済．
    """
    data = None

    if not reload and os.path.exists(filepath):
        try:
            data = pickle.load(open(filepath, 'rb'))
        except Exception:
            print(traceback.format_exc())

    if not data:
        data = load_data(filepath=orig_filepath, reload=reload)
        data = normalize(data)
        pickle.dump(data, open(filepath, 'wb'))

    return data

def make_info_df(data):
    key_list = []
    speaker_list = []
    utt_list = []
    emotion_list = []
    strength_list = []
    for key, x in data:
        key_list.append(key)
        speaker_list.append(key[:3])
        utt_list.append(key[3:7])
        emotion_list.append(key[7:10])
        strength_list.append(int(key[-1]))

    df = pd.DataFrame(
        data=dict(
            speaker=speaker_list,
            utt=utt_list,
            emotion=emotion_list,
            strength=strength_list
        ),
        index=key_list)
    
    return df


def make_info_df_with_train_info(data):
    info_df = make_info_df(data)

    # 正解ラベルを与える
    # 強度2,3はその感情，強度0はNEUとする.
    emotion_list = ['NEU'] + info_df['emotion'].unique().tolist()
    emo2no = dict([(emo , no) for (no, emo) in enumerate(emotion_list)])

    y_label = info_df['emotion'].copy()
    y_label[info_df['strength'] < 1] = 'NEU'
    y_label[info_df['strength'] == 1] = '*'
    y_label.name = 'y_label'

    y_number = y_label.apply(
        lambda label: emo2no[label] if label in emo2no else -1
    )
    y_number.name = 'y_number' 

    # 有効発話数
    num_effective_utterance = (y_number >= 0).sum()
    num_train_utterance = (num_effective_utterance * 15) // 20
    num_vali_utterance = (num_effective_utterance * 2) // 20

    idx_effective_utterance = info_df.index[(y_number >= 0)]
    idx_train = idx_effective_utterance[:num_train_utterance]
    idx_vali = idx_effective_utterance[num_train_utterance:(num_train_utterance + num_vali_utterance)]
    idx_test = idx_effective_utterance[(num_train_utterance + num_vali_utterance):]

    flag = pd.Series([False] * len(info_df), index=info_df.index)
    flag_train = flag.copy()
    flag_train[idx_train] = True
    flag_train.name = 'is_train'
    flag_vali = flag.copy()
    flag_vali[idx_vali] = True
    flag_vali.name = 'is_vali'
    flag_test = flag.copy()
    flag_test[idx_test] = True
    flag_test.name = 'is_test'

    return pd.concat([info_df, y_label, y_number, flag_train, flag_vali, flag_test], axis=1)


def save_info_df_to_csv(data=None, filename: str=INFO_CSV_PATH):
    if data is None:
        data = load_norm_data()

    df = make_info_df_with_train_info(data)
    df.to_csv(filename)
    import ipdb; ipdb.set_trace()


def build_data():
    data = load_norm_data()
    save_info_df_to_csv(data)

if __name__ == "__main__":
    build_data()
