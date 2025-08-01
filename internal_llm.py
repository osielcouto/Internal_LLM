"""
Avaliação de Bases de Dados Técnicas com Modelos de Linguagem de Grande Escala:  Um Estudo Aplicado ao Wi-Fi 7
Autor: Osiel do Couto Rosa

Este sistema permite:
1. Testar individualmente modelos LLM com acesso a documentos (RAG)
2. Comparar o desempenho de múltiplos modelos (com e sem RAG)
3. Executar baterias de testes automatizadas com lista de perguntas pré-definidas
4. Gerar relatórios com informações de conteúdo, fontes e tempo de resposta

Configurações principais podem ser ajustadas nas constantes no início do código.
"""

import time
from datetime import datetime
from tabulate import tabulate
import pandas as pd
import os
from typing import Tuple
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

# ==============================================
# CONFIGURAÇÕES PRINCIPAIS (AJUSTÁVEIS)
# ==============================================

# Diretório contendo os documentos em PDF
PDF_DIR = "./base_tcc"  # Caminho da pasta de documentos

# Nome do índice FAISS para armazenar os embeddings
FAISS_INDEX = "faiss_index"  # Pode ser alterado se necessário

# Modelos disponíveis e suas descrições
# Nota: "llama2_raw" é um termo especial para usar Llama 2 sem documentos
AVAILABLE_MODELS = {
    "mistral": "Mistral (4.4GB) - Melhor qualidade para tarefas complexas",
    "llama2": "Llama 2 (3.8GB) - Equilíbrio entre qualidade e desempenho",
    "tinyllama": "TinyLlama (637MB) - Leve e rápido, para testes rápidos",
    "llama2_raw": "Llama 2 Controle (3.8GB) - Sem documentos (usa Llama 2 puro)"
}

# Configurações de chunking para o processamento de documentos
CHUNK_SIZE = 500       # Tamanho dos pedaços de texto (em caracteres)
CHUNK_OVERLAP = 100    # Sobreposição entre chunks (para manter contexto)

# Modelo de embeddings
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"

# ==============================================
# FUNÇÕES PRINCIPAIS
# ==============================================

def select_model():
    """
    Permite ao usuário selecionar interativamente um modelo LLM.
    
    Retorna:
        str: Nome do modelo selecionado (chave em AVAILABLE_MODELS)
    
    Comportamento:
        - Exibe lista numerada de modelos disponíveis
        - Valida a entrada do usuário
        - Verifica se o modelo está instalado (exceto para llama2_raw)
        - Permite tentar novamente em caso de erro
    """
    print("\n🔍 Modelos LLM disponíveis:")
    for i, (name, desc) in enumerate(AVAILABLE_MODELS.items(), 1):
        print(f"{i}. {name.ljust(10)} - {desc}")
    
    while True:
        try:
            choice = int(input(f"\nSelecione o modelo (1-{len(AVAILABLE_MODELS)}): "))
            if 1 <= choice <= len(AVAILABLE_MODELS):
                selected = list(AVAILABLE_MODELS.keys())[choice-1]
                
                # Verificação especial para o modelo raw (não será checada a instalação devido ao termo especial)
                if selected == "llama2_raw":
                    return selected
                
                # Para outros modelos, verifica se está instalado
                try:
                    Ollama(model=selected).invoke("Teste")                                              # Método invoke envia uma solicitação para o modelo para verificar se está instalado
                    return selected
                except Exception:
                    print(f"❌ Modelo '{selected}' não encontrado. Baixe com: ollama pull {selected}")  # Se o invoke apresentar erro, solicita instalação via ollama
                    continue    
                    
            print("⚠️ Opção inválida!")
        except ValueError:
            print("⚠️ Digite apenas números!")

def load_documents():
    """
    Carrega e processa todos os documentos PDF do diretório configurado.
    
    Retorna:
        list: Lista de documentos carregados (langchain Document objects)
    
    Comportamento:
        - Varre o diretório PDF_DIR em busca de arquivos .pdf
        - Usa PyPDFLoader para extrair texto de cada página
        - Retorna lista consolidada de todos os documentos
        - Exibe erros individuais sem interromper o processo
    """
    print("\n📂 Carregando e processando documentos...")
    documents = []                                                      # Inicializa a lista de documentos
    for pdf_file in os.listdir(PDF_DIR):                                # Loop nos arquivos do diretório
        if pdf_file.endswith(".pdf"):                                   # Valida se são arquivos .pdf
            try:
                print(f"  - Processando: {pdf_file}")
                loader = PyPDFLoader(os.path.join(PDF_DIR, pdf_file))   # Extrai o conteúdo do PDF e retorna uma lista de objetos Document
                documents.extend(loader.load())                         # Adiciona os documentos à lista principal
            except Exception as e:
                print(f"❌ Erro no arquivo {pdf_file}: {str(e)}")
    return documents

def setup_rag_system(model_name):
    """
    Configura o sistema RAG completo para um modelo específico.
    
    Parâmetros:
        model_name (str): Nome do modelo a ser usado
    
    Retorna:
        tupla: (vectorstore, llm) - Componentes do sistema RAG
    
    Comportamento:
        - Para 'llama2_raw', retorna (None, None) pois não usa RAG
        - Processa documentos em chunks
        - Cria/recupera armazenamento vetorial FAISS
        - Configura o modelo LLM com parâmetros específicos
    """
    # Caso especial para o modelo raw (não configura RAG)
    if model_name == "llama2_raw":
        return None, None
    
    # 1. Processamento de documentos e dividindo em pedaços menores
    print("\n🔧 Configurando sistema RAG...")
    text_splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator="\n"
    )
    chunks = text_splitter.split_documents(load_documents())
    
    # 2. Configuração do armazenamento vetorial
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    
    # Tenta carregar índice existente ou cria novo
    if os.path.exists(FAISS_INDEX):
        print("  - Carregando índice FAISS existente...")
        vectorstore = FAISS.load_local(FAISS_INDEX, embeddings, allow_dangerous_deserialization=True)
    else:
        print("  - Criando novo índice FAISS...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(FAISS_INDEX)
    
    # 3. Configuração do modelo LLM
    print(f"  - Inicializando modelo {model_name}...")
    llm = Ollama(
        model=model_name,
        temperature=0.3,  # Controla a criatividade (0 = mais determinístico)
        system="""
        Você é um assistente acadêmico especializado em redes sem fio. Suas respostas devem ser em português do Brasil e baseadas 
        estritamente nos documentos fornecidos, com análise crítica e redação original. Responda com foco 
        em Wi-Fi 7 (IEEE 802.11be), explicando os conceitos de maneira clara, objetiva e tecnicamente 
        precisa, sem recorrer a conhecimento externo. Utilize apenas as informações presentes nos documentos, 
        mantendo o rigor técnico, evitando jargões excessivos e garantindo que a resposta seja interpretável 
        por profissionais da área técnica ou estudantes avançados.
        """,
        num_thread=4 if model_name != "tinyllama" else 2  # Ajuste de desempenho, pesquisas mostraram que usar o tinyllama com menos threads evita overhead e otimiza os resultados
    )
    
    return vectorstore, llm

def generate_response(question: str, vectorstore: FAISS, llm: Ollama) -> Tuple[str, list, float]:
    """
    Gera uma resposta usando o sistema RAG.
    
    Parâmetros:
        question (str): Pergunta do usuário
        vectorstore (FAISS): Armazenamento vetorial de documentos
        llm (Ollama): Modelo LLM configurado
    
    Retorna:
        tupla: (resposta, documentos_relevantes, tempo_execucao)
    
    Comportamento:
        - Busca trechos relevantes nos documentos
        - Constrói prompt contextualizado
        - Invoca o modelo LLM
        - Mede tempo de execução
    """
    start_time = time.time()                            # Inicia contagem de tempo
    
    # 1. Busca os trechos mais relevantes
    docs = vectorstore.similarity_search(question, k=5)  # Top 5 resultados(pode ser alterado)
    
    # 2. Constrói o contexto para o prompt
    context = "\n\n".join([
        f"Fonte: {os.path.basename(doc.metadata['source'])} (Página {doc.metadata.get('page', 'N/A')})\n" # Formata cada documento exibindo Fonte e Página
        f"Conteúdo: {doc.page_content}\n------"                                                           # Formata cada documento exibidno conteúdo e separa com ------
        for doc in docs
    ])
    
    # 3. Monta o prompt final
    prompt = f"""Com base nestes trechos:
    {context}
    Responda à pergunta abaixo com suas próprias palavras.
    Pergunta: {question}"""
    
    # 4. Invoca o modelo LLM enviando o contexto
    response = llm.invoke(prompt)
    elapsed_time = round(time.time() - start_time, 2)    # Finaliza contagem de tempo
    
    return response, docs, elapsed_time            

def generate_raw_response(question: str, model_name: str = "llama2") -> Tuple[str, float]:
    """
    Gera resposta SEM usar documentos (versão de controle do modelo).
    
    Parâmetros:
        question (str): Pergunta do usuário
        model_name (str): Nome do modelo a ser usado (padrão: "llama2")
    
    Retorna:
        tupla: (resposta, tempo_execucao)
    
    Comportamento:
        - Configura o modelo
        - Invoca o modelo sem contexto de documentos
        - Mede tempo de execução
    """
    start_time = time.time()                    # Inicia contagem de tempo
    
    llm = Ollama(
        model=model_name,
        temperature=0.3,
        system="""
        Você é um assistente técnico especializado em redes sem fio, com foco em Wi-Fi 7 (IEEE 802.11be). 
        Responda sempre em português do Brasil de forma clara, objetiva e acessível, utilizando termos corretos, mas explicando conceitos 
        de maneira simples. Evite jargões excessivos. Seja direto, didático e preciso, como se estivesse 
        explicando para um profissional da área técnica que deseja respostas rápidas, porém compreensíveis 
        por qualquer pessoa com conhecimento básico em tecnologia. Responda com conhecimento geral 
        (não use documentos específicos).
        """
    )
    
    response = llm.invoke(question)                        # Envia pergunta diretamente ao modelo sem contexto
    elapsed_time = round(time.time() - start_time, 2)      # Finaliza contagem de tempo
    
    return response, elapsed_time

# ==============================================
# MODOS DE OPERAÇÃO
# ==============================================

def benchmark_interativo():
    """
    Modo de benchmark interativo que testa todos os modelos com a mesma pergunta.
    
    Comportamento:
        - Permite ao usuário fazer perguntas iterativamente
        - Para cada pergunta, testa todos os modelos (com e sem RAG)
        - Exibe resultados comparativos em tempo real
        - Salva relatório CSV ao final
    """
    print("\n⚡ MODO BENCHMARK AUTOMÁTICO")
    print("Testará sequencialmente:\n"
          "1. Llama2 (CONTROLE)\n"
          "2. Mistral (RAG)\n"
          "3. Llama2 (RAG)\n"
          "4. TinyLlama (RAG)\n")
    
    # Pré-carrega todos os componentes para melhor desempenho
    print("\n📂 Carregando documentos...")
    documents = load_documents()
    
    print("🔧 Preparando sistema de embeddings...")
    text_splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator="\n"
    )
    chunks = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    
    print("🔧 Configurando vectorstore...")
    if os.path.exists(FAISS_INDEX):
        vectorstore = FAISS.load_local(FAISS_INDEX, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(FAISS_INDEX)
    
    # DataFrame para armazenar resultados para exportação posterior
    df_resultados = pd.DataFrame(columns=[
        "Modelo", 
        "Pergunta", 
        "Tempo(s)", 
        "Resposta",
        "Fontes"
    ])
    
    # Loop principal de perguntas
    while True:
        question = input("\n🔍 Digite a pergunta para comparar (ou 'sair'): ").strip()
        if question.lower() == 'sair':
            break
            
        print("\n⏳ Processando nos 4 modelos...")
        
        # 1. Testa versão de controle (sem documentos)
        try:
            print("  - Testando llama2 (CONTROLE)...", end='\r')
            response, tempo = generate_raw_response(question)
            df_resultados.loc[len(df_resultados)] = [
                "llama2 (CONTROLE)",
                question[:80] + ("..." if len(question) > 80 else ""),
                tempo,
                response.strip()[:500] + ("..." if len(response) > 500 else ""),
                "N/A (resposta controle)"
            ]
            print(f"✅ llama2 (CONTROLE)   | {str(tempo).ljust(5)}s")
        except Exception as e:
            print(f"❌ Falha em llama2 (CONTROLE): {str(e)}")
        
        # 2. Testa modelos com RAG
        modelos_rag = [
            ("mistral", "Mistral (RAG)"),
            ("llama2", "Llama2 (RAG)"),
            ("tinyllama", "TinyLlama (RAG)")
        ]
        
        for model_name, nome_exibicao in modelos_rag:
            try:
                print(f"  - Testando {nome_exibicao.ljust(12)}...", end='\r')
                
                # Configura LLM específico
                llm = Ollama(
                    model=model_name,
                    temperature=0.3,
                    system="""
        Você é um assistente acadêmico especializado em redes sem fio. Suas respostas devem ser em português do Brasil e baseadas 
        estritamente nos documentos fornecidos, com análise crítica e redação original. Responda com foco 
        em Wi-Fi 7 (IEEE 802.11be), explicando os conceitos de maneira clara, objetiva e tecnicamente 
        precisa, sem recorrer a conhecimento externo. Utilize apenas as informações presentes nos documentos, 
        mantendo o rigor técnico, evitando jargões excessivos e garantindo que a resposta seja interpretável 
        por profissionais da área técnica ou estudantes avançados.
        """,
                    num_thread=4 if model_name != "tinyllama" else 2
                )
                
                # Gera resposta RAG
                response, docs, tempo = generate_response(question, vectorstore, llm)
                fontes = "; ".join([
                    f"{os.path.basename(d.metadata['source'])} (p.{d.metadata.get('page', 'N/A')})" 
                    for d in docs
                ])
                
                # Armazena resultados usando Pandas
                df_resultados.loc[len(df_resultados)] = [
                    nome_exibicao,
                    question[:80] + ("..." if len(question) > 80 else ""),          
                    tempo,
                    response.strip()[:500] + ("..." if len(response) > 500 else ""),
                    fontes
                ]
                
                print(f"✅ {nome_exibicao.ljust(15)} | {str(tempo).ljust(5)}s")
                
            except Exception as e:
                print(f"❌ Falha em {nome_exibicao}: {str(e)}")
                continue
    
    # Salva e exibe resultados
    if not df_resultados.empty:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")                    # Obtém data e hora para nome do arquivo ser único
        nome_arquivo = f"benchmark_{timestamp}.csv"                             # Cria nome do arquivo_datahora
        df_resultados.to_csv(nome_arquivo, index=False, encoding='utf-8-sig')   # Exporta CSV 
        
        print("\n📊 RESULTADOS:")
        print(tabulate(df_resultados[["Modelo", "Tempo(s)"]],                   # Exibe resumo comparativo em tabela
                     headers="keys", 
                     tablefmt="simple"))
        
        print(f"\n💾 Relatório salvo como: {nome_arquivo}")                      # Informa ao usuário nome do arquivo salvo
    else:
        print("\n⚠️ Nenhum dado foi coletado")

def carregar_perguntas():
    """
    Carrega perguntas de um arquivo CSV com estrutura específica.
    
    Formato esperado:
        - Primeira linha: cabeçalho "Perguntas"
        - Linhas seguintes: perguntas (uma por linha)
        - Última linha: "/end" (opcional)
    
    Retorna:
        list: Lista de perguntas carregadas
    """
    try:
        with open('perguntas.csv', 'r', encoding='utf-8') as file:          # Nome do arquivo a ser lido e codificação para caracteres especiais
            cabecalho = file.readline().strip()
            if cabecalho != "Perguntas":                                    # Lê a primeira linha, remove espaços e verifica cabeçalho
                print(f"❌ Erro: Cabeçalho inválido. Esperado 'Perguntas', encontrado '{cabecalho}'")
                return []
            
            perguntas = [] 
            for linha in file:
                linha = linha.strip()                                       # Remove espaço em branco
                if linha == "/end":                                         # Para ao encontrar a flag "End"
                    break
                
                if linha:
                    perguntas.append(linha)                                 # Adiciona linha a lista de perguntas
            
            if not perguntas:
                print("❌ Erro: Nenhuma pergunta encontrada no arquivo")
                return []
            
            if len(perguntas) != 100:                                       # Valida se todas as perguntas foram carregadas
                print(f"⚠️ Aviso: Esperadas 100 perguntas, encontradas {len(perguntas)}")
            
            if linha != "/end":                                             # Verifica se flag estava no arquivo
                print("⚠️ Aviso: Flag /end não encontrada no final do arquivo")
            
            print(f"✅ {len(perguntas)} perguntas carregadas com sucesso")   # Informa total de perguntas carregadas com sucesso ao usuário
            return perguntas
            
    except FileNotFoundError:
        print("❌ Erro: Arquivo 'perguntas.csv' não encontrado")
        print("  Certifique-se que o arquivo está na mesma pasta do script")
        return []
    except Exception as e:
        print(f"❌ Erro inesperado ao carregar perguntas: {str(e)}")
        return []

def executar_bateria_testes(model_name):
    """
    Executa uma bateria de testes automatizada com perguntas pré-definidas.
    
    Parâmetros:
        model_name (str): Nome do modelo a ser testado
    
    Comportamento:
        - Carrega perguntas do arquivo
        - Para cada pergunta, gera resposta com o modelo selecionado
        - Armazena resultados detalhados
        - Gera relatório CSV ao final
    """
    perguntas = carregar_perguntas()
    if not perguntas:
        return
    
    # Configuração especial para o modelo raw
    if model_name == "llama2_raw":
        print(f"\n🔧 Iniciando teste com {model_name} (sem documentos) para {len(perguntas)} perguntas")
        
        resultados = []
        total_perguntas = len(perguntas)
        
        for i, pergunta in enumerate(perguntas, 1):               # Percorre lista de perguntas
            print(f"\n📝 Progresso: {i}/{total_perguntas}")       # Exibe progresso
            print(f"   Pergunta: {pergunta[:70]}...")             # Exibe trecho da pergunta limitando a 70 caracteres
            
            try:
                resposta, tempo = generate_raw_response(pergunta)
                
                resultados.append({                                 # Armazena resultados
                    'Modelo': model_name,
                    'Pergunta': pergunta,
                    'Tempo(s)': tempo,
                    'Resposta': resposta[:2000],
                    'Fontes': "N/A (resposta controle)"
                })
                
                print(f"   ✅ Respondido em {resultados[-1]['Tempo(s)']}s")  # Exibe tempo de resposta
                
            except Exception as e:                                  # Captura erros caso ocorram
                print(f"   ❌ Erro: {str(e)}")
                resultados.append({
                    'Modelo': model_name,
                    'Pergunta': pergunta,
                    'Tempo(s)': 0,
                    'Resposta': f"ERRO: {str(e)}",
                    'Fontes': "N/A"
                })
    else:
        # Configuração normal para modelos com RAG
        vectorstore, llm = setup_rag_system(model_name)
        print(f"\n🔧 Iniciando teste com {model_name} para {len(perguntas)} perguntas")
        
        resultados = []
        total_perguntas = len(perguntas)
        
        for i, pergunta in enumerate(perguntas, 1):
            print(f"\n📝 Progresso: {i}/{total_perguntas}")
            print(f"   Pergunta: {pergunta[:70]}...")
            
            try:
                start_time = time.time()
                resposta, docs, tempo = generate_response(pergunta, vectorstore, llm)
                
                fontes = "; ".join(
                    f"{os.path.basename(d.metadata['source'])} (p.{d.metadata.get('page', 'N/A')})"
                    for d in docs
                )
                
                resultados.append({
                    'Modelo': model_name,
                    'Pergunta': pergunta,
                    'Tempo(s)': round(time.time() - start_time, 2),
                    'Resposta': resposta[:2000],
                    'Fontes': fontes
                })
                
                print(f"   ✅ Respondido em {resultados[-1]['Tempo(s)']}s")
                
            except Exception as e:
                print(f"   ❌ Erro: {str(e)}")
                resultados.append({
                    'Modelo': model_name,
                    'Pergunta': pergunta,
                    'Tempo(s)': 0,
                    'Resposta': f"ERRO: {str(e)}",
                    'Fontes': "N/A"
                })
    
    # Salva resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    nome_arquivo = f"resultados_{model_name}_{timestamp}.csv"
    
    pd.DataFrame(resultados).to_csv(nome_arquivo, index=False, encoding='utf-8-sig')
    print(f"\n💾 Arquivo salvo: {nome_arquivo}")
    print(f"📊 Total de perguntas respondidas: {len([r for r in resultados if not r['Resposta'].startswith('ERRO')])}/{total_perguntas}")

# ==============================================
# Função principal
# ==============================================

def main():
    """
    Função principal que gerencia o menu interativo.
    
    Oferece três modos de operação:
    1. Modo individual: Testa um modelo por vez com perguntas interativas
    2. Benchmark comparativo: Compara todos os modelos com a mesma pergunta
    3. Bateria de testes: Executa perguntas pré-definidas em um modelo específico
    """
    print("\n" + "="*50)
    print("SISTEMA DE COMPARAÇÃO DE MODELOS LLM".center(50))
    print("="*50)
    print("\nSelecione o modo de operação:")
    print("1. Modo normal (uso individual)")
    print("2. Benchmark comparativo (testar todos os modelos)")
    print("3. Bateria de testes")
    print("4. Sair")
    
    while True:
        escolha = input("\nDigite o modo desejado (1/2/3/4): ").strip()
        
        # Modo individual
        if escolha == "1":
            print("\n🛠️  Configuração do Sistema")
            model_name = select_model()                 # Validação e seleção do modelo 
            
            # Caso especial para o modelo raw
            if model_name == "llama2_raw":
                print(f"\n✅ Sistema pronto! Modelo: {model_name} (sem documentos)")
                print("🔍 Digite suas perguntas ou 'sair' para voltar ao menu.")
                
                while True:
                    question = input("\n❓ Pergunta: ").strip()
                    if question.lower() == 'sair':      # Converte todas as strings para minuscula e compara conteúdo
                        break
                        
                    try:
                        print("\n⏳ Processando...")
                        response, time_sec = generate_raw_response(question)
                        
                        print(f"\n💡 Resposta ({time_sec}s):")
                        print(response)
                        
                    except Exception as e:
                        print(f"\n❌ Erro: {str(e)}")
                        print("Dica: Verifique se o servidor Ollama está rodando ('ollama serve')")
            else:
                # Configuração normal para modelos com RAG
                vectorstore, llm = setup_rag_system(model_name)
                print(f"\n✅ Sistema pronto! Modelo: {model_name}")
                print("🔍 Digite suas perguntas ou 'sair' para voltar ao menu.")
                
                while True:
                    question = input("\n❓ Pergunta: ").strip()
                    if question.lower() == 'sair':
                        break
                        
                    try:
                        print("\n⏳ Processando...")
                        response, docs, time_sec = generate_response(question, vectorstore, llm)
                        
                        print(f"\n💡 Resposta ({time_sec}s):")
                        print(response)
                        
                        print("\n📚 Fontes utilizadas:")
                        for doc in docs:
                            print(f"- {os.path.basename(doc.metadata['source'])} (página {doc.metadata.get('page', 'N/A')})")
                            
                    except Exception as e:
                        print(f"\n❌ Erro: {str(e)}")
                        print("Dica: Verifique se o servidor Ollama está rodando ('ollama serve')")
            
        # Benchmark comparativo
        elif escolha == "2":
            benchmark_interativo()
            
        # Bateria de testes
        elif escolha == "3":
            model_name = select_model()
            try:
                executar_bateria_testes(model_name)
            except Exception as e:
                print(f"\n❌ Falha durante os testes: {str(e)}")
                if model_name != "llama2_raw":
                    print("Possíveis causas:")
                    print("- Documentos não encontrados no diretório especificado")
                    print("- Índice FAISS corrompido")
                    print("- Problema de conexão com o Ollama")
                
        # Sair
        elif escolha == "4":
            print("\n👋 Encerrando o programa...")
            break
        else:
            print("⚠️ Opção inválida! Digite um número entre 1 e 4.")

if __name__ == "__main__":
    # Suprime avisos de depreciação
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
    
    # Verifica se o diretório de documentos existe
    if not os.path.exists(PDF_DIR):
        print(f"❌ Erro: Diretório de documentos '{PDF_DIR}' não encontrado!")
        print("  Crie o diretório ou ajuste a variável PDF_DIR no código.")
        exit(1)
    

    main()
