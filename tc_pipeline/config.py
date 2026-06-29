"""
tc_pipeline/config.py
─────────────────────
Configuración centralizada del pipeline de scraping.

Reemplaza las variables globales de scraper.py (URL_API, HEADERS,
CARPETA_RAIZ_DESCARGAS, MAX_WORKERS, etc.) con un dataclass inmutable
que sirve como única fuente de verdad para todos los módulos.

Uso:
    from tc_pipeline.config import PipelineConfig
    config = PipelineConfig()                        # valores por defecto
    config = PipelineConfig(max_workers=20)          # override parcial
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────
# Códigos HTTP que disparan reintentos automáticos
# ─────────────────────────────────────────────────────────────────────────
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class PipelineConfig:
    """Configuración inmutable del pipeline de scraping del Tribunal Constitucional.

    Attributes:
        api_url: Endpoint de búsqueda cronológica de sentencias.
        user_agent: User-Agent para las peticiones HTTP.
        accept_header: Accept header para las peticiones HTTP.
        connect_timeout: Timeout de conexión en segundos (aplica a API y PDFs).
        api_read_timeout: Timeout de lectura para llamadas a la API.
        pdf_read_timeout: Timeout de lectura para descarga de PDFs.
        max_retries: Número máximo de reintentos por petición fallida.
        retry_base_delay: Delay base en segundos para backoff exponencial (1s, 2s, 4s, 8s).
        page_delay: Pausa entre páginas de la API para no saturar el servidor.
        retryable_status_codes: Códigos HTTP que disparan reintentos.
        download_root: Directorio raíz para PDFs descargados.
        max_workers: Número de workers concurrentes para descargas.
        manifest_db: Ruta a la base de datos SQLite del manifiesto.
    """

    # ── API del Tribunal Constitucional ──────────────────────────────────
    api_url: str = (
        "https://jurisbackend.sedetc.gob.pe"
        "/api/visitor/sentencia/busqueda/cronologico"
    )
    api_url_avanzada: str = (
        "https://jurisbackend.sedetc.gob.pe"
        "/api/visitor/sentencia/busqueda/avanzada"
    )
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    accept_header: str = "application/json"

    # ── Timeouts (segundos) ──────────────────────────────────────────────
    # Separados para fallar rápido en conexión y tolerar PDFs grandes.
    connect_timeout: float = 5.0
    api_read_timeout: float = 20.0
    pdf_read_timeout: float = 45.0

    # ── Política de reintentos ───────────────────────────────────────────
    max_retries: int = 10
    retry_base_delay: float = 5.0
    page_delay: float = 2.5
    retryable_status_codes: frozenset[int] = field(
        default_factory=lambda: RETRYABLE_STATUS_CODES
    )

    # ── Descargas (legacy) ───────────────────────────────────────────────
    download_root: Path = Path("EXPEDIENTES")
    max_workers: int = 10

    # ── Nuevas rutas de datos (pipeline CSV) ─────────────────────────────
    csv_output_root: Path = Path("data/csv")
    sentencia_raw_root: Path = Path("data/sentencia-raw")
    auto_resolucion_raw_root: Path = Path("data/auto-resolucion-raw")
    sentencia_extract_root: Path = Path("data/sentencia-Extract")
    auto_resolucion_extract_root: Path = Path("data/auto-resolucion-Extract")

    # ── Extracción de PDF ────────────────────────────────────────────────
    pdf_extraction_timeout: float = 30.0  # Timeout por PDF individual (segundos)

    # ── Manifiesto SQLite ────────────────────────────────────────────────
    manifest_db: Path = Path("data/manifests/pipeline_state.db")

    # ── Helpers ──────────────────────────────────────────────────────────

    @property
    def headers(self) -> dict[str, str]:
        """Headers HTTP para todas las peticiones."""
        return {
            "User-Agent": self.user_agent,
            "Accept": self.accept_header,
        }

    @property
    def api_timeout(self) -> tuple[float, float]:
        """Tupla (connect, read) para peticiones a la API."""
        return (self.connect_timeout, self.api_read_timeout)

    @property
    def pdf_timeout(self) -> tuple[float, float]:
        """Tupla (connect, read) para descarga de PDFs."""
        return (self.connect_timeout, self.pdf_read_timeout)


# ─────────────────────────────────────────────────────────────────────────────
# Configuración del Pipeline de Machine Learning
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MLConfig:
    """Configuración inmutable del pipeline de entrenamiento XGBoost.

    Centraliza todos los hiperparámetros del clasificador multiclase, la
    conectividad a ChromaDB y los parámetros de reproducibilidad del
    experimento. Actúa como única fuente de verdad para los módulos
    ``data_loader``, ``model_trainer`` y ``model_evaluator``.

    Attributes:
        chroma_db_path: Ruta al directorio de almacenamiento persistente de ChromaDB.
        chroma_collection_name: Nombre de la colección vectorial a consultar.
        embedding_model_name: Modelo SentenceTransformer utilizado durante la indexación.
        metadata_label_key: Clave del metadato que contiene la etiqueta objetivo.
        metadata_groupby_key: Clave del metadato para agrupar chunks por documento.
        test_size: Proporción del conjunto de prueba (0.0–1.0).
        random_state: Semilla global para reproducibilidad.
        models_output_dir: Directorio de salida para artefactos entrenados.
        xgb_n_estimators: Número de árboles (rondas de boosting).
        xgb_max_depth: Profundidad máxima de cada árbol base.
        xgb_learning_rate: Tasa de aprendizaje (eta) del boosting.
        xgb_subsample: Fracción de muestras usadas por árbol.
        xgb_colsample_bytree: Fracción de features usadas por árbol.
        xgb_objective: Función objetivo para clasificación multiclase.
        xgb_eval_metric: Métrica de evaluación durante el entrenamiento.
        xgb_use_label_encoder: Deshabilita el codificador interno (gestionamos con sklearn).
        xgb_tree_method: Motor de árboles; 'hist' es el más eficiente en CPU.
        xgb_n_jobs: Workers en paralelo (-1 = todos los núcleos disponibles).
    """

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    chroma_db_path: str = "data/chroma_storage"
    chroma_collection_name: str = "jurisprudencia_tc"
    embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # ── Metadatos del dataset ────────────────────────────────────────────────
    metadata_label_key: str = "sentido_resolucion"
    metadata_groupby_key: str = "numero_expediente"

    # ── Partición y reproducibilidad ─────────────────────────────────────────
    test_size: float = 0.20
    random_state: int = 42

    # ── Artefactos del modelo ────────────────────────────────────────────────
    models_output_dir: Path = Path("models")

    # ── Hiperparámetros XGBoost ──────────────────────────────────────────────
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_objective: str = "multi:softprob"
    xgb_eval_metric: str = "mlogloss"
    xgb_use_label_encoder: bool = False
    xgb_tree_method: str = "hist"
    xgb_n_jobs: int = -1

    # ── Helpers ──────────────────────────────────────────────────────────────

    @property
    def model_artifact_path(self) -> Path:
        """Ruta completa al archivo JSON del modelo XGBoost serializado."""
        return self.models_output_dir / "xgb_classifier.json"

    @property
    def encoder_artifact_path(self) -> Path:
        """Ruta completa al archivo joblib del LabelEncoder serializado."""
        return self.models_output_dir / "label_encoder.joblib"

    @property
    def logreg_model_artifact_path(self) -> Path:
        """Ruta completa al archivo joblib del modelo Regresión Logística serializado."""
        return self.models_output_dir / "logreg_classifier.joblib"

    @property
    def logreg_encoder_artifact_path(self) -> Path:
        """Ruta completa al archivo joblib del LabelEncoder de Regresión Logística."""
        return self.models_output_dir / "logreg_label_encoder.joblib"

    @property
    def svm_model_artifact_path(self) -> Path:
        """Ruta completa al archivo joblib del modelo SVM serializado."""
        return self.models_output_dir / "svm_classifier.joblib"

    @property
    def svm_encoder_artifact_path(self) -> Path:
        """Ruta completa al archivo joblib del LabelEncoder de SVM."""
        return self.models_output_dir / "svm_label_encoder.joblib"

    @property
    def xgb_params(self) -> dict:
        """Diccionario de hiperparámetros listo para pasarse al XGBClassifier."""
        return {
            "n_estimators": self.xgb_n_estimators,
            "max_depth": self.xgb_max_depth,
            "learning_rate": self.xgb_learning_rate,
            "subsample": self.xgb_subsample,
            "colsample_bytree": self.xgb_colsample_bytree,
            "objective": self.xgb_objective,
            "eval_metric": self.xgb_eval_metric,
            "use_label_encoder": self.xgb_use_label_encoder,
            "tree_method": self.xgb_tree_method,
            "n_jobs": self.xgb_n_jobs,
            "random_state": self.random_state,
        }
