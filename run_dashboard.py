"""
🚀 GUIA RÁPIDO DE EXECUÇÃO - Dashboard de Saúde

Execute este arquivo para ver instruções passo a passo de como rodar o dashboard.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(text):
    """Imprime um cabeçalho formatado"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")

def print_step(number, text):
    """Imprime um passo numerado"""
    print(f"  {number}️⃣  {text}")

def check_python_version():
    """Verifica se a versão do Python é adequada"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor} detectado (OK)")
        return True
    else:
        print(f"❌ Python 3.8+ necessário. Versão atual: {version.major}.{version.minor}")
        return False

def check_requirements():
    """Verifica se as dependências estão instaladas"""
    try:
        import streamlit
        import sqlalchemy
        import pandas
        print("✅ Dependências principais instaladas (OK)")
        return True
    except ImportError as e:
        print(f"❌ Dependência faltando: {e}")
        return False

def install_requirements():
    """Instala as dependências"""
    print_header("📥 INSTALANDO DEPENDÊNCIAS")
    print("  Aguarde enquanto as dependências são instaladas...\n")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("\n✅ Dependências instaladas com sucesso!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Erro ao instalar dependências")
        return False

def run_dashboard():
    """Executa o dashboard Streamlit"""
    print_header("🚀 INICIANDO DASHBOARD")
    print("  O navegador será aberto automaticamente em: http://localhost:8501\n")
    print("  Pressione CTRL+C para parar o servidor.\n")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"])
    except KeyboardInterrupt:
        print("\n\n✅ Dashboard encerrado com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao executar dashboard: {e}")

def main():
    """Função principal"""
    
    print_header("💪 DASHBOARD DE SAÚDE E ROTINA")
    
    print("  Bem-vindo ao Dashboard de Saúde!")
    print("  Este script o ajudará a configurar e executar a aplicação.\n")
    
    # Verificar Python
    print_header("1️⃣  VERIFICANDO AMBIENTE")
    if not check_python_version():
        print("  ⚠️  Por favor, instale Python 3.8 ou superior")
        sys.exit(1)
    
    # Verificar requirements
    print("\n  Verificando dependências...")
    if not check_requirements():
        print("\n  As dependências não estão instaladas.")
        response = input("\n  Deseja instalar agora? (s/n): ").strip().lower()
        if response == 's':
            if not install_requirements():
                sys.exit(1)
        else:
            print("  ⚠️  Dependências necessárias não estão instaladas")
            print("  Execute: pip install -r requirements.txt")
            sys.exit(1)
    
    # Mostrar funcionalidades
    print_header("2️⃣  FUNCIONALIDADES DISPONÍVEIS")
    print("  📝 Formulário - Registre seus dados de saúde diários")
    print("  📈 Dashboard - Visualize gráficos e estatísticas")
    print("  📋 Histórico - Consulte todos os registros com filtros")
    print("  💾 Banco de Dados - Armazenamento local em SQLite")
    
    # Informações adicionais
    print_header("3️⃣  INFORMAÇÕES DO PROJETO")
    
    db_path = Path("health_database.db")
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        print(f"  📁 Banco de Dados: {db_path.name} ({size_mb:.2f} MB)")
    else:
        print(f"  📁 Banco de Dados: Será criado automaticamente")
    
    print(f"  📂 Diretório: {Path.cwd()}")
    print(f"  🐍 Python: {sys.version.split()[0]}")
    
    # Iniciar dashboard
    print_header("4️⃣  INICIANDO APLICAÇÃO")
    response = input("  Deseja iniciar o Dashboard agora? (s/n): ").strip().lower()
    
    if response == 's':
        run_dashboard()
    else:
        print("\n  Para iniciar o Dashboard, execute:")
        print("  $ streamlit run dashboard.py\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✅ Execução cancelada pelo usuário")
        sys.exit(0)
