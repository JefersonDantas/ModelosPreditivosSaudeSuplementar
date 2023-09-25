# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 15:34:30 2023

@author: jeferson.dantas
Objetivo: Coletar arquivos na ANS dos meses indicados e descompactar adicionado na pasta de dados brutos ans
"""

import os
import requests
from bs4 import BeautifulSoup
import zipfile

# %% baixar arquivos
UF = 'SE'

# URL base
BASE_URL = 'https://dadosabertos.ans.gov.br/FTP/PDA/informacoes_consolidadas_de_beneficiarios/'

# Pasta local para salvar os arquivos
SAVE_PATH = 'arquivosanszip'  # Altere para o caminho desejado

# Crie a pasta se ela não existir
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Lista de subdiretórios
SUBDIRECTORIES = ['201405', '201406', '201407', '201408', '201409', '201410', '201411', '201412'
                  ,'201501','201502','201503','201504','201505', '201506', '201507', '201508', '201509', '201510', '201511', '201512'
                  ,'201601','201602','201603','201604','201605', '201606', '201607', '201608', '201609', '201610', '201611', '201612'
                  ,'201701','201702','201703','201704','201705', '201706', '201707', '201708', '201709', '201710', '201711', '201712'
                  ,'201801','201802','201803','201804','201805', '201806', '201807', '201808', '201809', '201810', '201811', '201812'
                  ,'201901','201902','201903','201904','201905', '201906', '201907', '201908', '201909', '201910', '201911', '201912'
                  ,'202001','202002','202003','202004','202005', '202006', '202007', '202008', '202009', '202010', '202011', '202012'
                  ,'202101','202102','202103','202104','202105', '202106', '202107', '202108', '202109', '202110', '202111', '202112'
                  ,'202201','202202','202203','202204','202205', '202206', '202207', '202208', '202209', '202210', '202211', '202212'
                  ,'202301','202302','202303','202304','202305', '202306'
                  ]  

# Crie a pasta se ela não existir
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Função para baixar o arquivo
def download_file(url, save_path):
    response = requests.get(url, stream=True)
    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

# Para cada subdiretório na lista, acesse a URL e baixe os arquivos .zip com '_SE' no nome
for subdirectory in SUBDIRECTORIES:
    zip_file = 'ben'+subdirectory+'_'+UF+'.zip'
    subfolder_url = BASE_URL + subdirectory+'/'+zip_file
    save_file_path = os.path.join(SAVE_PATH, zip_file)
    download_file(subfolder_url, save_file_path)
    print(f"Arquivo {zip_file} baixado com sucesso!")

print("Todos os arquivos foram baixados!")

# %% descompactar arquivos

# Pasta onde os arquivos descompactados serão armazenados
DESTINO_PATH = 'dados brutos ans'  # Altere para o caminho onde você deseja armazenar os arquivos descompactados

# Crie a pasta de destino se ela não existir
if not os.path.exists(DESTINO_PATH):
    os.makedirs(DESTINO_PATH)

# Função para descompactar um arquivo
def unzip_file(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# Itere sobre todos os arquivos na pasta SAVE_PATH
for filename in os.listdir(SAVE_PATH):
    if filename.endswith('.zip'):
        file_path = os.path.join(SAVE_PATH, filename)
        unzip_file(file_path, DESTINO_PATH)
        print(f"Arquivo {filename} descompactado com sucesso!")

print("Todos os arquivos foram descompactados!")