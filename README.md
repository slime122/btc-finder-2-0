# BTC Finder 2.0

Cliente Python/PyOpenCL com painel Web para varredura de ranges da Solo Pool BTC Puzzle.

O projeto atual substitui o fork Node.js antigo por uma arquitetura pequena:
cliente REST, orquestrador Python e kernel OpenCL C para secp256k1.


YleapgtcCstbbHdKYNQsAddNnNKAwHhDZJrtWMxMumFfcMIKfSeAlytuPjBpgiXpMbbgMsbBBCafyttgSqXIrfxEejdHVrldGxSMEphgeIUMpnpZDXbXwVDIPqCASaBr

## Estrutura

```text
btc-finder-2-0/
├── btc_finder/
│   ├── __init__.py
│   ├── api.py
│   ├── opencl_kernel.py
│   └── solver.py
├── kernels/
│   └── secp256k1.cl
├── web/
│   └── static/
│       └── index.html
├── .gitignore
├── main.py
├── package.json
├── README.md
└── requirements.txt
```

## Requisitos

- Python 3.10+
- OpenCL disponível no sistema
- macOS, Linux ou Windows

No macOS com Intel Iris 640, use lotes pequenos para evitar o watchdog da GPU.
O padrão atual é `4096` chaves por disparo de kernel.

## Instalação

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Execução

Subir o painel web:

```bash
python3 main.py
```

ou:

```bash
npm run start
```

Abra `http://127.0.0.1:3000`. O worker não inicia automaticamente; informe `UserToken`,
`WorkerName`, hardware e batch size, depois pressione `START`.

## CLI Direto

Para executar sem painel web, use `--no-web`. Defina seu token da pool:

```bash
export BTCPUZZLE_USER_TOKEN="seu-token"
export BTCPUZZLE_WORKER_NAME="macbook"
```

Rodar Puzzle 71 na GPU com lote seguro:

```bash
python3 main.py --no-web --puzzle 71 --batch-size 256
```

Forçar CPU OpenCL:

```bash
python3 main.py --no-web --puzzle 71 --cpu --batch-size 4096
```

Testar no Puzzle 38:

```bash
python3 main.py --no-web --puzzle 38 --batch-size 256
```

Os mesmos comandos estão disponíveis via npm como atalhos:

```bash
npm run start
npm run start:cpu
npm run test:p38
```

## Observações

- `FOUND_KEY.txt` é gerado somente se a chave principal for encontrada e fica fora do Git.
- `--batch-size 0` usa o padrão automático: GPU Intel integrada usa `256`; CPU usa `4096`.
- `--batch-size` pode ser aumentado manualmente, mas valores altos podem acionar `GPU hang occurred` no macOS.
- A API usada é `https://api.btcpuzzle.info`.
