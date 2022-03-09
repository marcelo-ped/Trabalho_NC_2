## Este é o trabalho 1 de Computação Natural de Marcelo Pedro.

## Instalar:
Para baixar o código e o relatório desse trabalho basta dá o comando num terminal linux:

```bash
gh repo clone marcelo-ped/Trabalho_NC_2
```

Caso não tenha o git instalado na sua máquina dê o comando abaixo:

```bash
sudo apt-get install git
```
Para instalar os pacotes necessários deste trabalho basta dá o comando:

```bash
pip install -r requirements.txt
```

Esse trabalho foi executado em python na versão 3.6, caso você tenha algum problema com a instalação, instalar a versão 3.6 pode resolver o problema. Para isso basta dar os seguintes comando em um terminal:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.6
```
 
## Como executar o código deste trabalho

Para execução do código deste trabalho basta executar o programa main.py setando 2 flags cujo nomes são dataset e algorithm. A flag dataset aceita 3 valores, digite 0 para executar o código com o conjunto de dados de íris, digite 1 para executar o código com o conjunto de dados wine e digite 2 para executar o código com o conjunto de dados breast cancer. A flag algorithm 2 valores, digite 0 para executar o algoritmo PSO, digite 1 para executar o algoritmo GA.

Um exemplo de comando para executar o código deste trabalho é mostrado abaixo:

```bash
python3 main.py --algorithm 1 --dataset 1
```

