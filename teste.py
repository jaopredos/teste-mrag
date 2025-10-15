import torch
import chromadb
from colpali_engine.models import ColQwen2, ColQwen2Processor
from transformers.utils.import_utils import is_flash_attn_2_available
import glob
from PIL import Image
import os

MODEL_NAME = "vidore/colqwen2-v1.0"
COLLECTION_NAME = "documentos_teste"
IMAGES_PATH = "imagens_extraidas"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
ATTN_IMPL = "flash_attention_2" if is_flash_attn_2_available() else None

model = ColQwen2.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
    attn_implementation=ATTN_IMPL,
).eval()
processor = ColQwen2Processor.from_pretrained(MODEL_NAME)

client = chromadb.Client()
collection = client.get_or_create_collection(name=COLLECTION_NAME)


def load_document_images_from_folder(folder_path):
    print(f"Buscando imagens na pasta: {folder_path}")
    page_data = []

    # Busca por todos os arquivos de imagem suportados
    image_paths = glob.glob(os.path.join(folder_path, "*.png"))
    image_paths.extend(glob.glob(os.path.join(folder_path, "*.jpg")))
    image_paths.extend(glob.glob(os.path.join(folder_path, "*.jpeg")))
    image_paths.sort() 
    
    if not image_paths:
        raise FileNotFoundError(f"Nenhuma imagem (.png, .jpg ou .jpeg) encontrada em {folder_path}")

    for img_path in image_paths:
        # Carrega a imagem
        image = Image.open(img_path).convert("RGB")
        
        page_data.append({
            "image": image, 
            "source_file": os.path.basename(img_path),
            "page_id": os.path.splitext(os.path.basename(img_path))[0] # ID baseado no nome do arquivo (sem extensão)
        })

    return page_data

try:
    pages_data = load_document_images_from_folder(IMAGES_PATH)
except FileNotFoundError as e:
    print(f"Erro Fatal: {e}")
    exit()

print(f"Iniciando indexação de {len(pages_data)} páginas...")

with torch.no_grad():
    for data in pages_data:
        image = data["image"]
        page_id = data["page_id"]

        print(f"Processando: {data['source_file']}")
        
        batch_images = processor.process_images([image]).to(model.device)

        # B. Gerar os Embeddings Multi-Vetoriais
        image_embeddings_output = model(**batch_images)
        token_embeddings = image_embeddings_output['document_embeddings'].squeeze(0) 

        # C. Ajuste para ChromaDB (Média dos Vetores)
        # O embedding que será salvo no banco de dados.
        avg_embedding = token_embeddings.mean(dim=0).cpu().numpy().tolist()

        # D. Armazenar no ChromaDB
        # NOTA: O campo 'documents' do ChromaDB está sendo preenchido com o nome do arquivo, 
        # pois não temos o texto limpo para passar ao LLM mais tarde.
        collection.add(
            embeddings=[avg_embedding],
            documents=[f"Página de Documento Visual: {data['source_file']}"], 
            metadatas={
                "source_file": data['source_file'], 
                "page_id": page_id, 
                "model": MODEL_NAME
            },
            ids=[page_id]
        )

print("\n✅ Indexação concluída.")
print(f"Total de páginas indexadas: {collection.count()}")