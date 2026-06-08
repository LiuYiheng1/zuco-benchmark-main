import numpy as np
import os

def analyze_data():
    print('='*70)
    print('ZuCo 2.0 数据集详细分析报告')
    print('='*70)
    
    # 1. 查看 EEG 特征
    print('\n' + '='*70)
    print('1. EEG 电极特征 (electrode_features_all)')
    print('='*70)
    eeg_data = np.load('features/YAC_electrode_features_all.npy', allow_pickle=True).item()
    print(f'   数据类型: dict')
    print(f'   样本数量: {len(eeg_data)}')
    
    keys = list(eeg_data.keys())
    first_sample = eeg_data[keys[0]]
    print(f'   key 格式: {keys[0]} (被试_条件_句子ID_索引)')
    print(f'   特征维度: {len(first_sample)} (420个电极 + 1个标签)')
    print(f'   值类型: {type(first_sample[0])}')
    print(f'   标签位置: 最后一位')
    print(f'   标签值: {first_sample[-1]}')
    
    # 统计标签
    nr_count = sum(1 for k in eeg_data.keys() if '_NR_' in k)
    tsr_count = sum(1 for k in eeg_data.keys() if '_TSR_' in k)
    print(f'\\n   标签分布:')
    print(f'     NR (正常阅读): {nr_count} ({nr_count/len(eeg_data)*100:.1f}%)')
    print(f'     TSR (语义违背): {tsr_count} ({tsr_count/len(eeg_data)*100:.1f}%)')
    
    # 2. 查看 Gaze 特征
    print('\n' + '='*70)
    print('2. Gaze 眼动特征 (sent_gaze_sacc)')
    print('='*70)
    gaze_data = np.load('features/YAC_sent_gaze_sacc.npy', allow_pickle=True).item()
    print(f'   数据类型: dict')
    print(f'   样本数量: {len(gaze_data)}')
    
    gaze_keys = list(gaze_data.keys())
    first_gaze = gaze_data[gaze_keys[0]]
    print(f'   key 格式: {gaze_keys[0]}')
    print(f'   特征维度: {len(first_gaze)} (9个眼动特征 + 1个标签)')
    
    gaze_features = [
        'fixation_number', 'omission_rate', 'reading_speed',
        'mean_sacc_amp', 'mean_sacc_dur', 'mean_sacc_velocity',
        'max_sacc_amp', 'max_sacc_dur', 'max_sacc_velocity', 'label'
    ]
    print('\\n   特征顺序:')
    for i, feat in enumerate(gaze_features):
        print(f'     [{i}] {feat}: {first_gaze[i]}')
    
    # 3. 查看频段特征
    print('\n' + '='*70)
    print('3. EEG 频段特征')
    print('='*70)
    
    bands = ['theta', 'alpha', 'beta', 'gamma']
    for band in bands:
        data = np.load(f'features/YAC_{band}_mean.npy', allow_pickle=True)
        print(f'   {band}_mean.npy:')
        print(f'     类型: {type(data)}')
        print(f'     形状: {data.shape if hasattr(data, "shape") else "scalar"}')
        if hasattr(data, '__len__'):
            if len(data) > 0 and isinstance(data[0], (int, float)):
                print(f'     值范围: [{data.min():.3f}, {data.max():.3f}]')
    
    # 4. 查看阅读速度
    print('\n' + '='*70)
    print('4. 阅读速度特征 (reading_speed)')
    print('='*70)
    rs = np.load('features/YAC_reading_speed.npy', allow_pickle=True)
    print(f'   类型: {type(rs)}')
    print(f'   值: {rs}')
    
    # 5. 被试信息
    print('\n' + '='*70)
    print('5. 被试分类')
    print('='*70)
    print('   Y-前缀: Young Adult Control (年轻成人对照组)')
    print('   X-前缀: Developmental Dyslexia (发展性阅读障碍组)')
    
    # 统计被试数量
    all_files = os.listdir('features')
    eeg_files = [f for f in all_files if 'electrode_features_all' in f]
    y_subjects = [f[:3] for f in eeg_files if f.startswith('Y')]
    x_subjects = [f[:3] for f in eeg_files if f.startswith('X')]
    
    print(f'\\n   总被试数: {len(eeg_files)}')
    print(f'   Y组(对照组): {len(y_subjects)} 人 ({", ".join(sorted(set(y_subjects)))})')
    print(f'   X组(障碍组): {len(x_subjects)} 人 ({", ".join(sorted(set(x_subjects)))})')
    
    # 6. 数据对齐问题说明
    print('\n' + '='*70)
    print('6. 关键数据对齐问题')
    print('='*70)
    print('   ⚠️ 重要发现: gaze 文件中同一个句子有两个条目')
    print('   示例: YAC_NR_0_0 和 YAC_TSR_0_250')
    print('   这意味着同一个句子在两种条件下都被记录了')
    print('   必须使用 label + sentence_id 同时匹配才能正确对齐')
    
    # 验证这一点
    same_sentence_nr = [k for k in gaze_keys if '_NR_0_' in k]
    same_sentence_tsr = [k for k in gaze_keys if '_TSR_0_' in k]
    print(f'\\n   句子 0 的 gaze 条目:')
    print(f'     NR 条件: {same_sentence_nr}')
    print(f'     TSR 条件: {same_sentence_tsr}')

if __name__ == '__main__':
    analyze_data()