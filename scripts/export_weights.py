import torch
import os

REMOTE_MODEL_ID = 'fancyovo/XingLing-Chat-0.68B-SFT'
REMOTE_MODEL_PATH = 'model_sft.pth'
MODEL_PATH = 'data/'
EXPORT_PATH = 'data/model'

def export_weights(model_path, export_path):
    '''
    将模型model_path(.pth文件)中的权重以嵌套文件夹形式导出到export_path下
    '''
    state_dict = torch.load(model_path, map_location='cpu')

    for key, tensor in state_dict.items():
        parts = key.split('.')
        sub_dir = os.path.join(export_path, *parts[:-1])
        file_name = parts[-1] + '.bin'
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        data = tensor.detach().cpu().numpy().astype('float32')
        full_path = os.path.join(sub_dir, file_name)
        with open(full_path, 'wb') as f:
            f.write(data.tobytes())

        print(f'Export {key} | shape:{tensor.shape} -> {full_path}')


def download_model(model_path):
    '''
    从魔塔社区上下载模型权重到model_path
    '''
    from modelscope.hub.file_download import model_file_download
    model_dir = model_file_download(
        model_id = REMOTE_MODEL_ID, 
        file_path = REMOTE_MODEL_PATH, 
        local_dir = model_path, 
    )
    print(f'模型已从魔塔社区上下载至 {model_dir}')

if __name__ == '__main__':
    download_model(MODEL_PATH)
    export_weights(os.path.join(MODEL_PATH, REMOTE_MODEL_PATH), EXPORT_PATH)
    print(f'模型权重已全部导出至 {EXPORT_PATH}')