"""
📊 Script para Popular Banco de Dados com Dados de Exemplo
Use este script para testar o dashboard com dados reais
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import random

# Adicionar o diretório ao path
sys.path.insert(0, str(Path(__file__).parent))

from dashboard import get_db_session, HealthRecord, Base, engine

def create_sample_data():
    """Cria dados de exemplo para testes"""
    
    # Criar tabelas se não existirem
    Base.metadata.create_all(bind=engine)
    
    db = get_db_session()
    
    # Verificar se já existem dados
    if db.query(HealthRecord).count() > 0:
        response = input(
            "⚠️  Banco de dados já contém dados. "
            "Deseja adicionar mais registros? (s/n): "
        ).strip().lower()
        if response != 's':
            print("❌ Operação cancelada")
            db.close()
            return
    
    print("📝 Gerando dados de exemplo...\n")
    
    # Configurações para diferentes perfis
    profiles = {
        "Sedentário": {
            "passos_range": (2000, 5000),
            "sono_range": (6.0, 7.5),
            "treino_probs": [0.7, 0.2, 0.1],  # [Nenhum, Leve, Intenso]
            "kcal_range": (1800, 2200),
        },
        "Normal": {
            "passos_range": (6000, 10000),
            "sono_range": (7.0, 8.0),
            "treino_probs": [0.3, 0.5, 0.2],
            "kcal_range": (2000, 2500),
        },
        "Muito Ativo": {
            "passos_range": (10000, 15000),
            "sono_range": (7.5, 8.5),
            "treino_probs": [0.1, 0.3, 0.6],
            "kcal_range": (2300, 3000),
        }
    }
    
    # Gerar 30 dias de dados (3 por perfil)
    base_date = datetime.now() - timedelta(days=30)
    records_created = 0
    
    for i in range(30):
        current_date = base_date + timedelta(days=i)
        
        # Selecionar perfil aleatoriamente
        perfil = random.choice(list(profiles.keys()))
        config = profiles[perfil]
        
        # Gerar valores
        passos = random.randint(config["passos_range"][0], config["passos_range"][1])
        sono_horas = round(random.uniform(config["sono_range"][0], config["sono_range"][1]), 1)
        humor = random.randint(0, 2)
        
        # Treino baseado em probabilidades
        treino = random.choices([0, 1, 2], weights=config["treino_probs"])[0]
        
        kcal = random.randint(config["kcal_range"][0], config["kcal_range"][1])
        
        # Macronutrientes proporcionais às calorias
        carboidrato = int((kcal * 0.45) / 4)  # 45% de carbs, 4 kcal/g
        proteina = int((kcal * 0.25) / 4)      # 25% de proteína
        gordura = int((kcal * 0.30) / 9)       # 30% de gordura, 9 kcal/g
        
        agua_ml = random.randint(1500, 3000)
        deficit_kcal = random.randint(-500, 500)
        
        # Criar registro
        record = HealthRecord(
            data=current_date + timedelta(hours=random.randint(20, 23)),
            perfil=perfil,
            passos=passos,
            sono_horas=sono_horas,
            humor=humor,
            kcal=kcal,
            carboidrato=carboidrato,
            proteina=proteina,
            gordura=gordura,
            agua_ml=agua_ml,
            treino=treino,
            deficit_kcal=deficit_kcal,
        )
        
        db.add(record)
        records_created += 1
        
        # Mostrar progresso
        perfil_emoji = {
            "Sedentário": "🪑",
            "Normal": "🚶",
            "Muito Ativo": "💪"
        }[perfil]
        
        treino_text = ["Nenhum", "Leve", "Intenso"][treino]
        humor_text = ["Bom 😊", "Neutro 😐", "Ruim 😢"][humor]
        
        print(f"  {i+1:2d}. {perfil_emoji} {perfil:12s} | Passos: {passos:5d} | "
              f"Treino: {treino_text:7s} | Humor: {humor_text}")
    
    # Salvar no banco
    try:
        db.commit()
        db.close()
        print(f"\n✅ {records_created} registros criados com sucesso!")
        print(f"📊 O Dashboard agora tem dados para visualizar gráficos e estatísticas")
    except Exception as e:
        db.rollback()
        db.close()
        print(f"❌ Erro ao salvar: {e}")

def clear_database():
    """Limpa todos os dados do banco"""
    response = input(
        "⚠️  AVISO: Isto deletará todos os registros. Deseja continuar? (s/n): "
    ).strip().lower()
    
    if response != 's':
        print("❌ Operação cancelada")
        return
    
    try:
        db = get_db_session()
        db.query(HealthRecord).delete()
        db.commit()
        db.close()
        print("✅ Banco de dados limpo com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao limpar: {e}")

def show_stats():
    """Mostra estatísticas do banco de dados"""
    try:
        db = get_db_session()
        count = db.query(HealthRecord).count()
        db.close()
        
        if count == 0:
            print("📭 Nenhum registro encontrado")
        else:
            print(f"📊 Total de registros: {count}")
            print(f"📅 Aproximadamente {count // 30:.1f} meses de dados" if count >= 30 else 
                  f"📅 {count} dias de dados")
    except Exception as e:
        print(f"❌ Erro ao verificar: {e}")

def main():
    """Menu principal"""
    print("\n" + "=" * 60)
    print("  📊 GERENCIADOR DE DADOS - Dashboard de Saúde")
    print("=" * 60 + "\n")
    
    print("  Opções:")
    print("  1️⃣  Criar dados de exemplo (30 dias)")
    print("  2️⃣  Ver estatísticas do banco")
    print("  3️⃣  Limpar banco de dados")
    print("  4️⃣  Sair")
    print()
    
    choice = input("  Escolha uma opção (1-4): ").strip()
    
    if choice == '1':
        print()
        create_sample_data()
    elif choice == '2':
        print()
        show_stats()
    elif choice == '3':
        print()
        clear_database()
    elif choice == '4':
        print("\n✅ Até logo!\n")
        sys.exit(0)
    else:
        print("\n❌ Opção inválida\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✅ Operação cancelada")
        sys.exit(0)
