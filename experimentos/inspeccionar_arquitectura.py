"""Inspecciona la red y muestra ventanas de Conv1D con datos artificiales."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.conv_softmax import Conv1DSoftmaxNet


def inspeccionar() -> dict:
    torch.manual_seed(42)
    modelo = Conv1DSoftmaxNet(num_features=15, num_classes=3).eval()
    formas = {}
    capas = {
        "conv_1": modelo.backbone.features.blocks[0],
        "conv_2": modelo.backbone.features.blocks[1],
        "flatten": modelo.backbone.features.flatten,
        "mlp": modelo.backbone.mlp,
        "salida": modelo.head,
    }
    hooks = [capa.register_forward_hook(
        lambda _modulo, _entradas, salida, nombre=nombre: formas.update({nombre: list(salida.shape)})
    ) for nombre, capa in capas.items()]
    entrada = torch.arange(30, dtype=torch.float32).reshape(2, 15) / 30
    with torch.no_grad():
        salida = modelo(entrada)
        salida_canal = modelo(entrada.unsqueeze(1))
    for hook in hooks:
        hook.remove()

    parametros = {nombre: sum(p.numel() for p in capa.parameters()) for nombre, capa in capas.items()}

    # Una ventana ilustrativa: entrada 0..14, kernel [1,2,3], padding 1.
    secuencia = torch.arange(15, dtype=torch.float32).reshape(1, 1, 15)
    kernel = torch.tensor([[[1.0, 2.0, 3.0]]])
    ventanas = F.pad(secuencia, (1, 1)).unfold(2, 3, 1)
    respuesta = F.conv1d(secuencia, kernel, padding=1)

    # Dos convoluciones lineales de kernel 3: impulso central y soporte observado.
    impulso = torch.zeros(1, 1, 15)
    impulso[0, 0, 7] = 1
    unos = torch.ones(1, 1, 3)
    dos_capas = F.conv1d(F.conv1d(impulso, unos, padding=1), unos, padding=1)
    posiciones = torch.where(dos_capas[0, 0] != 0)[0].tolist()

    return {
        "entrada": list(entrada.shape),
        "entrada_con_canal": list(entrada.unsqueeze(1).shape),
        "formas": formas,
        "parametros_por_capa": parametros,
        "parametros_totales": sum(p.numel() for p in modelo.parameters()),
        "salidas_finitas": bool(torch.isfinite(salida).all()),
        "entradas_equivalentes_en_evaluacion": bool(torch.allclose(salida, salida_canal)),
        "ejemplo_convolucion": {
            "kernel": kernel.flatten().tolist(),
            "posicion": 7,
            "ventana": ventanas[0, 0, 7].tolist(),
            "salida": respuesta[0, 0, 7].item(),
            "ventana_borde": ventanas[0, 0, 0].tolist(),
        },
        "dos_convoluciones_posiciones_afectadas_por_impulso_central": posiciones,
    }


if __name__ == "__main__":
    print(json.dumps(inspeccionar(), ensure_ascii=False, indent=2))
