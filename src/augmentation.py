import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import zoom

def jittering(window, sigma=0.01):
    """
    Adiciona ruído gaussiano (simulando microestrutura de mercado e spread).
    """
    noise = np.random.normal(0, sigma, window.shape)
    return window + noise

def magnitude_warping(window, sigma=0.1, knot=4):
    """
    Distorce magnitude usando splines (simulando ciclos de maior/menor volatilidade).
    """
    orig_steps = np.arange(window.shape[0])
    random_warps = np.random.normal(1.0, sigma, (knot + 2,))
    warp_steps = np.linspace(0, window.shape[0] - 1, num=knot + 2)
    warper = CubicSpline(warp_steps, random_warps)(orig_steps)
    # Multiplica toda a janela (todas as features) pela curva
    return window * warper[:, np.newaxis]

def window_slicing(window, slice_ratio=0.9):
    """
    Fatia a janela aleatoriamente e redimensiona para tamanho original via interpolação.
    Evita memorização de posições fixas de padrões.
    """
    target_len = int(window.shape[0] * slice_ratio)
    if target_len >= window.shape[0]:
        return window
        
    start = np.random.randint(0, window.shape[0] - target_len)
    sliced = window[start:start + target_len]
    
    # Redimensiona para (32, 6) novamente
    resized = zoom(sliced, (window.shape[0] / target_len, 1), order=1)
    
    # Ajuste de tamanho caso o zoom gere +/- 1 índice de diferença devido a arredondamento
    if resized.shape[0] > window.shape[0]:
        resized = resized[:window.shape[0]]
    elif resized.shape[0] < window.shape[0]:
        # Preenche com último valor se faltar (edge case raro do zoom)
        diff = window.shape[0] - resized.shape[0]
        padding = np.repeat(resized[-1:], diff, axis=0)
        resized = np.vstack([resized, padding])
        
    return resized
