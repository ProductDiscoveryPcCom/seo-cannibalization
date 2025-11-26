"""
SEO Cannibalization Analyzer v3.1
Herramienta profesional de análisis de canibalizaciones SEO
Enfoque: Análisis por FAMILIA + Arquitectura óptima + Recomendaciones priorizadas

CORRECCIONES v3.1:
- Clasificación mejorada usando archivo de categorías
- Caching para mejor rendimiento
- Eliminados returns problemáticos en tabs
- Árbol dinámico basado en datos reales
- % recuperación configurable
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from urllib.parse import urlparse
import requests
from io import StringIO
import json
import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

# ==================== CONFIGURACIÓN ====================
st.set_page_config(
    page_title="SEO Cannibalization Analyzer v3",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS profesional y accesible
st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; max-width: 1400px; }
    
    /* Score cards */
    .score-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    .score-value { font-size: 2.5rem; font-weight: 700; line-height: 1; }
    .score-label { font-size: 0.85rem; color: #64748b; margin-top: 0.5rem; }
    .score-good { color: #16a34a; }
    .score-warning { color: #d97706; }
    .score-bad { color: #dc2626; }
    
    /* Tree structure */
    .tree-container {
        font-family: 'Courier New', Consolas, monospace;
        font-size: 13px;
        line-height: 1.6;
        background: #1e293b;
        color: #e2e8f0;
        padding: 1.5rem;
        border-radius: 8px;
        overflow-x: auto;
        white-space: pre;
    }
    .tree-plp { color: #38bdf8; font-weight: 600; }
    .tree-pdp { color: #a78bfa; }
    .tree-blog { color: #4ade80; }
    .tree-problem { color: #f87171; }
    .tree-ok { color: #4ade80; }
    .tree-comment { color: #64748b; font-style: italic; }
    
    /* Cards */
    .card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }
    .card-danger { border-left: 4px solid #dc2626; background: #fef2f2; }
    .card-warning { border-left: 4px solid #d97706; background: #fffbeb; }
    .card-success { border-left: 4px solid #16a34a; background: #f0fdf4; }
    .card-info { border-left: 4px solid #0284c7; background: #eff6ff; }
    
    /* Badges */
    .badge { display: inline-block; padding: 3px 10px; border-radius: 4px; font-size: 11px; font-weight: 600; }
    .badge-plp { background: #0284c7; color: white; }
    .badge-pdp { background: #7c3aed; color: white; }
    .badge-blog { background: #059669; color: white; }
    .badge-critical { background: #7f1d1d; color: white; }
    .badge-high { background: #dc2626; color: white; }
    .badge-medium { background: #d97706; color: white; }
    .badge-low { background: #16a34a; color: white; }
    
    /* URL display */
    .url-slug {
        font-family: 'Courier New', monospace;
        font-size: 13px;
        color: #0f172a;
        background: #f1f5f9;
        padding: 4px 8px;
        border-radius: 4px;
        word-break: break-all;
    }
    .url-slug a { color: #0284c7; text-decoration: none; }
    .url-slug a:hover { text-decoration: underline; }
    
    /* Query tag */
    .query-tag {
        display: inline-block;
        background: #dbeafe;
        color: #1e40af;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 12px;
        margin: 2px 0;
    }
    
    /* Metrics inline */
    .metrics-inline {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-top: 0.5rem;
        font-size: 12px;
        color: #64748b;
    }
    .metrics-inline strong { color: #0f172a; }
    
    /* Health bar */
    .health-bar {
        height: 8px;
        background: #e2e8f0;
        border-radius: 4px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .health-bar-fill {
        height: 100%;
        border-radius: 4px;
    }
    
    /* Rec item */
    .rec-item {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .rec-impact {
        display: inline-block;
        background: #fef3c7;
        color: #92400e;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ==================== ENUMS Y DATACLASSES ====================
class UrlType(Enum):
    PLP = "PLP"
    PDP = "PDP"
    BLOG = "BLOG"
    OTHER = "OTHER"

class Severity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

@dataclass
class FamilyHealth:
    family: str
    total_urls: int
    plp_count: int
    pdp_count: int
    blog_count: int
    total_clicks: int
    total_impressions: int
    cannibalized_queries: int
    cannibalized_clicks: int
    health_score: float
    main_plp: str
    issues: List[str]

@dataclass 
class Cannibalization:
    query: str
    family: str
    urls: List[Dict]
    severity: Severity
    total_clicks: int
    total_impressions: int
    click_concentration: float
    position_spread: float
    main_url: Dict
    competing_urls: List[Dict]
    issue_type: str
    recommendation: str
    recoverable_clicks: int


# ==================== FUNCIONES DE CLASIFICACIÓN ====================
def extract_slug(url: str) -> str:
    """Extrae el slug desde el dominio"""
    try:
        parsed = urlparse(url)
        return parsed.path.strip('/') or '/'
    except:
        return str(url)

def get_known_plp_slugs(categories_df: pd.DataFrame) -> Set[str]:
    """Extrae slugs conocidos de PLPs del archivo de categorías"""
    known_slugs = set()
    if categories_df is not None and not categories_df.empty:
        # Buscar columna de slug
        slug_col = None
        for col in categories_df.columns:
            if col.lower().strip() in ['slug', 'url', 'categoria', 'category']:
                slug_col = col
                break
        
        if slug_col:
            for val in categories_df[slug_col].dropna():
                # Extraer solo el slug si es URL completa
                if 'http' in str(val):
                    slug = extract_slug(str(val))
                else:
                    slug = str(val).strip('/').lower()
                if slug:
                    known_slugs.add(slug)
    return known_slugs

@st.cache_data
def classify_url_with_context(url: str, known_plps: tuple) -> Tuple[str, str, int]:
    """
    Clasifica URL según estructura PCComponentes usando contexto de PLPs conocidas.
    
    Reglas PCComponentes:
    1. Si el slug está en la lista de categorías conocidas → PLP
    2. Si tiene /blog/, /noticias/, /guia/ → BLOG  
    3. Si slug en raíz con 1 segmento y MUCHOS guiones (5+) → PDP (producto)
    4. Si slug en raíz con 1 segmento y POCOS guiones → PLP (categoría)
    5. Si múltiples segmentos con / → PLP (subcategoría)
    
    Returns: (tipo, familia, profundidad)
    """
    known_plps_set = set(known_plps) if known_plps else set()
    
    try:
        parsed = urlparse(url)
        path = parsed.path.lower().strip('/')
        
        if not path:
            return 'PLP', 'home', 0
        
        segments = [s for s in path.split('/') if s]
        first_segment = segments[0] if segments else ''
        
        # 1. Verificar si está en PLPs conocidas
        if path in known_plps_set or first_segment in known_plps_set:
            family = first_segment.replace('-', ' ')
            return 'PLP', family, len(segments)
        
        # 2. Detectar BLOG por patrones en URL
        blog_patterns = ['blog', 'noticias', 'guia', 'guias', 'magazine', 'revista', 'articulo']
        if any(p in path for p in blog_patterns):
            # Extraer familia del blog
            for seg in segments:
                if seg not in blog_patterns:
                    return 'BLOG', seg.replace('-', ' '), len(segments)
            return 'BLOG', 'general', len(segments)
        
        # 3. Un solo segmento en raíz
        if len(segments) == 1:
            slug = segments[0]
            hyphen_count = slug.count('-')
            
            # PDP: slug largo con muchos guiones (familia-marca-modelo-specs-atributos)
            # Típicamente: portatil-asus-rog-strix-g15-ryzen-9-rtx-4070 = 9 guiones
            # Una PLP como "tarjetas-graficas" solo tiene 1 guión
            if hyphen_count >= 5:
                # La familia es la primera palabra
                family = slug.split('-')[0]
                return 'PDP', family, 1
            else:
                # PLP de categoría (portatiles, monitores, tarjetas-graficas)
                return 'PLP', slug.replace('-', ' '), 1
        
        # 4. Múltiples segmentos con /
        # Estructura: /categoria/subcategoria/ o /categoria/producto
        family = first_segment.replace('-', ' ')
        last_segment = segments[-1]
        
        # Si el último segmento tiene muchos guiones, probablemente es PDP
        if last_segment.count('-') >= 5:
            return 'PDP', family, len(segments)
        
        # Si no, es PLP/subcategoría
        return 'PLP', family, len(segments)
        
    except Exception:
        return 'OTHER', 'unknown', 0


# ==================== LECTURA DE CSV ====================
@st.cache_data
def read_csv_flexible(content: bytes, name: str = "archivo", skip_comments: bool = False):
    """Lee CSV con detección automática de formato"""
    try:
        # Decodificar
        try:
            text = content.decode('utf-8')
        except:
            try:
                text = content.decode('latin-1')
            except:
                text = content.decode('cp1252')
        
        lines = text.split('\n')
        
        # Saltar comentarios si es necesario
        skip_rows = 0
        if skip_comments:
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                if (line_stripped.startswith('#') or 
                    line_stripped.startswith('=') or 
                    line_stripped == '' or
                    'Report Suite' in line or
                    'Date Range' in line):
                    skip_rows = i + 1
                else:
                    break
        
        # Detectar separador
        data_line = lines[skip_rows] if skip_rows < len(lines) else lines[0]
        if '\t' in data_line:
            sep = '\t'
        elif ';' in data_line and data_line.count(';') > data_line.count(','):
            sep = ';'
        else:
            sep = ','
        
        # Leer con pandas
        from io import StringIO
        df = pd.read_csv(
            StringIO(text), 
            sep=sep, 
            skiprows=skip_rows,
            on_bad_lines='skip'
        )
        df.columns = df.columns.str.strip()
        return df
        
    except Exception as e:
        st.error(f"❌ Error al leer {name}: {str(e)}")
        return None


# ==================== ANÁLISIS POR FAMILIA ====================
def analyze_family_health(df: pd.DataFrame, family: str) -> Optional[FamilyHealth]:
    """Analiza la salud SEO de una familia de productos"""
    
    # Filtrar por familia (más robusto)
    family_lower = family.lower()
    mask = (
        (df['family'].str.lower() == family_lower) |
        (df['slug'].str.lower().str.startswith(family_lower.replace(' ', '-'))) |
        (df['slug'].str.lower().str.contains(f"^{family_lower.replace(' ', '-')}[/-]", regex=True, na=False))
    )
    family_df = df[mask].copy()
    
    if len(family_df) == 0:
        return None
    
    # Contar tipos
    plp_count = len(family_df[family_df['url_type'] == 'PLP'])
    pdp_count = len(family_df[family_df['url_type'] == 'PDP'])
    blog_count = len(family_df[family_df['url_type'] == 'BLOG'])
    
    # Métricas de tráfico
    total_clicks = int(family_df['top_query_clicks'].sum())
    total_impressions = int(family_df['top_query_impressions'].sum())
    
    # Detectar canibalizaciones
    query_counts = family_df.groupby('top_query').size()
    cannibalized_queries = int((query_counts > 1).sum())
    
    cannib_queries = query_counts[query_counts > 1].index.tolist()
    cannib_df = family_df[family_df['top_query'].isin(cannib_queries)]
    cannibalized_clicks = int(cannib_df['top_query_clicks'].sum())
    
    # PLP principal
    plps = family_df[family_df['url_type'] == 'PLP'].sort_values('top_query_clicks', ascending=False)
    main_plp = plps.iloc[0]['slug'] if len(plps) > 0 else "No hay PLP"
    
    # Issues
    issues = []
    if plp_count == 0:
        issues.append("❌ No hay PLP principal para esta familia")
    if plp_count > 3:
        issues.append(f"⚠️ Demasiadas PLPs ({plp_count}) - posible fragmentación")
    if cannibalized_queries > 5:
        issues.append(f"🔴 {cannibalized_queries} queries con múltiples URLs compitiendo")
    if blog_count > 0 and plp_count > 0:
        blog_queries = set(family_df[family_df['url_type'] == 'BLOG']['top_query'].dropna())
        plp_queries = set(family_df[family_df['url_type'] == 'PLP']['top_query'].dropna())
        overlap = blog_queries & plp_queries
        if overlap:
            issues.append(f"🔴 {len(overlap)} queries donde Blog compite con PLP")
    if cannibalized_clicks > total_clicks * 0.3 and total_clicks > 0:
        pct = cannibalized_clicks / total_clicks * 100
        issues.append(f"⚠️ {pct:.0f}% del tráfico afectado por canibalizaciones")
    
    # Health score
    health_score = 100
    if plp_count == 0:
        health_score -= 30
    if cannibalized_queries > 0:
        health_score -= min(30, cannibalized_queries * 2)
    if total_clicks > 0 and cannibalized_clicks > total_clicks * 0.3:
        health_score -= 20
    if len(issues) > 3:
        health_score -= 10
    health_score = max(0, min(100, health_score))
    
    return FamilyHealth(
        family=family,
        total_urls=len(family_df),
        plp_count=plp_count,
        pdp_count=pdp_count,
        blog_count=blog_count,
        total_clicks=total_clicks,
        total_impressions=total_impressions,
        cannibalized_queries=cannibalized_queries,
        cannibalized_clicks=cannibalized_clicks,
        health_score=health_score,
        main_plp=main_plp,
        issues=issues
    )


def detect_cannibalizations(df: pd.DataFrame, family: str, min_clicks: int, recovery_pct: float) -> List[Cannibalization]:
    """Detecta canibalizaciones dentro de una familia"""
    
    family_lower = family.lower()
    mask = (
        (df['family'].str.lower() == family_lower) |
        (df['slug'].str.lower().str.startswith(family_lower.replace(' ', '-'))) |
        (df['slug'].str.lower().str.contains(f"^{family_lower.replace(' ', '-')}[/-]", regex=True, na=False))
    )
    family_df = df[mask].copy()
    
    cannibalizations = []
    
    for query, group in family_df.groupby('top_query'):
        if pd.isna(query) or not str(query).strip() or len(group) < 2:
            continue
        
        total_clicks = int(group['top_query_clicks'].sum())
        if total_clicks < min_clicks:
            continue
        
        group = group.sort_values('top_query_clicks', ascending=False)
        
        urls_data = []
        for _, row in group.iterrows():
            urls_data.append({
                'url': row['url'],
                'slug': row['slug'],
                'type': row['url_type'],
                'clicks': int(row['top_query_clicks']),
                'impressions': int(row['top_query_impressions']),
                'position': float(row['top_query_position']) if pd.notna(row['top_query_position']) else 0,
                'top_query': str(row['top_query'])
            })
        
        main_url = urls_data[0]
        competing = urls_data[1:]
        
        total_impressions = int(group['top_query_impressions'].sum())
        click_concentration = (main_url['clicks'] / total_clicks * 100) if total_clicks > 0 else 0
        
        positions = [u['position'] for u in urls_data if u['position'] > 0]
        position_spread = max(positions) - min(positions) if len(positions) > 1 else 0
        
        # Tipo de issue
        types_involved = set(u['type'] for u in urls_data)
        
        if 'BLOG' in types_involved and 'PLP' in types_involved:
            issue_type = "Blog vs PLP"
            recommendation = "Redirigir blog a /blog/{familia}/ o añadir canonical hacia PLP"
        elif 'PDP' in types_involved and 'PLP' in types_involved:
            issue_type = "PDP vs PLP"
            recommendation = "Añadir canonical en PDP hacia PLP padre"
        elif types_involved == {'PLP'}:
            issue_type = "PLPs duplicadas"
            recommendation = "Consolidar en una PLP principal y 301 el resto"
        elif types_involved == {'PDP'}:
            issue_type = "PDPs compitiendo"
            recommendation = "Diferenciar contenido o consolidar si son duplicados"
        elif types_involved == {'BLOG'}:
            issue_type = "Blogs duplicados"
            recommendation = "Fusionar en un único artículo y 301 el resto"
        else:
            issue_type = "Mixto"
            recommendation = "Analizar caso por caso"
        
        # Severidad
        url_count = len(urls_data)
        if url_count >= 4 or (issue_type == "Blog vs PLP" and click_concentration < 60):
            severity = Severity.CRITICAL
        elif url_count >= 3 or issue_type in ["Blog vs PLP", "PLPs duplicadas"]:
            severity = Severity.HIGH
        elif click_concentration < 70:
            severity = Severity.MEDIUM
        else:
            severity = Severity.LOW
        
        # Clics recuperables
        dispersed = total_clicks - main_url['clicks']
        recoverable = int(dispersed * recovery_pct)
        
        cannibalizations.append(Cannibalization(
            query=str(query),
            family=family,
            urls=urls_data,
            severity=severity,
            total_clicks=total_clicks,
            total_impressions=total_impressions,
            click_concentration=click_concentration,
            position_spread=position_spread,
            main_url=main_url,
            competing_urls=competing,
            issue_type=issue_type,
            recommendation=recommendation,
            recoverable_clicks=recoverable
        ))
    
    # Ordenar por impacto
    severity_order = {Severity.CRITICAL: 4, Severity.HIGH: 3, Severity.MEDIUM: 2, Severity.LOW: 1}
    cannibalizations.sort(key=lambda x: (severity_order[x.severity], x.recoverable_clicks), reverse=True)
    
    return cannibalizations


# ==================== ÁRBOL DE ARQUITECTURA ====================
def generate_architecture_tree(df: pd.DataFrame, family: str, cannibalizations: List[Cannibalization]) -> Tuple[str, str]:
    """Genera árboles de arquitectura actual y recomendada basados en datos reales"""
    
    family_lower = family.lower()
    family_slug = family.replace(' ', '-')
    
    mask = (
        (df['family'].str.lower() == family_lower) |
        (df['slug'].str.lower().str.startswith(family_slug))
    )
    family_df = df[mask].copy()
    
    if len(family_df) == 0:
        return "No hay datos", "No hay datos"
    
    # Obtener queries con problemas
    problem_queries = {c.query for c in cannibalizations}
    
    # Organizar URLs por tipo
    plps = family_df[family_df['url_type'] == 'PLP'].sort_values('top_query_clicks', ascending=False)
    pdps = family_df[family_df['url_type'] == 'PDP'].sort_values('top_query_clicks', ascending=False)
    blogs = family_df[family_df['url_type'] == 'BLOG'].sort_values('top_query_clicks', ascending=False)
    
    # ========== ÁRBOL ACTUAL ==========
    lines = []
    lines.append(f'<span class="tree-plp">📁 /{family_slug}/</span>')
    lines.append(f'<span class="tree-comment">   Estructura actual · {len(family_df)} URLs</span>')
    lines.append('')
    
    # PLPs
    if len(plps) > 0:
        lines.append('<span class="tree-plp">├── 📂 PLPs ({} categorías)</span>'.format(len(plps)))
        for i, (_, row) in enumerate(plps.head(6).iterrows()):
            is_last = i == min(len(plps) - 1, 5)
            prefix = '│   └──' if is_last else '│   ├──'
            slug_short = row['slug'][:35] + '...' if len(str(row['slug'])) > 35 else row['slug']
            
            has_problem = row['top_query'] in problem_queries
            if has_problem:
                lines.append(f'<span class="tree-problem">{prefix} ⚠️ /{slug_short}</span>')
                lines.append(f'<span class="tree-comment">│       └─ Canibaliza: "{str(row["top_query"])[:25]}..."</span>')
            else:
                lines.append(f'<span class="tree-ok">{prefix} ✓ /{slug_short}</span>')
        
        if len(plps) > 6:
            lines.append(f'<span class="tree-comment">│   └── ... +{len(plps) - 6} PLPs más</span>')
    else:
        lines.append('<span class="tree-problem">├── ❌ No hay PLPs detectadas</span>')
    
    lines.append('│')
    
    # PDPs
    if len(pdps) > 0:
        problem_pdps = pdps[pdps['top_query'].isin(problem_queries)]
        ok_pdps_count = len(pdps) - len(problem_pdps)
        
        lines.append('<span class="tree-pdp">├── 📦 PDPs ({} productos)</span>'.format(len(pdps)))
        
        for i, (_, row) in enumerate(problem_pdps.head(4).iterrows()):
            slug_short = row['slug'][:30] + '...' if len(str(row['slug'])) > 30 else row['slug']
            lines.append(f'<span class="tree-problem">│   ├── ⚠️ /{slug_short}</span>')
            lines.append(f'<span class="tree-comment">│   │   └─ Rankea por query genérica</span>')
        
        if ok_pdps_count > 0:
            lines.append(f'<span class="tree-ok">│   └── ✓ {ok_pdps_count} PDPs sin conflicto</span>')
    else:
        lines.append('<span class="tree-pdp">├── 📦 No hay PDPs</span>')
    
    lines.append('│')
    
    # Blogs
    if len(blogs) > 0:
        lines.append('<span class="tree-blog">└── 📝 BLOG ({} posts)</span>'.format(len(blogs)))
        for i, (_, row) in enumerate(blogs.head(4).iterrows()):
            is_last = i == min(len(blogs) - 1, 3)
            prefix = '    └──' if is_last else '    ├──'
            slug_short = row['slug'][:30] + '...' if len(str(row['slug'])) > 30 else row['slug']
            
            has_problem = row['top_query'] in problem_queries
            if has_problem:
                lines.append(f'<span class="tree-problem">{prefix} ⚠️ /{slug_short}</span>')
                lines.append(f'<span class="tree-comment">        └─ Compite con PLP transaccional</span>')
            else:
                lines.append(f'<span class="tree-ok">{prefix} ✓ /{slug_short}</span>')
    else:
        lines.append('<span class="tree-blog">└── 📝 No hay posts de blog</span>')
    
    current_tree = '\n'.join(lines)
    
    # ========== ÁRBOL RECOMENDADO ==========
    rec_lines = []
    rec_lines.append(f'<span class="tree-plp">📁 /{family_slug}/</span> <span class="tree-comment">← PLP Principal</span>')
    rec_lines.append(f'<span class="tree-comment">   Target: "{family}", "comprar {family}"</span>')
    rec_lines.append('')
    
    # Subcategorías basadas en PLPs existentes
    rec_lines.append('<span class="tree-plp">├── 📂 Subcategorías recomendadas</span>')
    
    # Detectar posibles subcategorías de las PLPs existentes
    if len(plps) > 0:
        for i, (_, row) in enumerate(plps.head(4).iterrows()):
            slug = str(row['slug'])
            if '/' in slug:
                subcat = slug.split('/')[-1]
                rec_lines.append(f'<span class="tree-plp">│   ├── /{family_slug}/{subcat}/</span>')
    
    rec_lines.append(f'<span class="tree-plp">│   ├── /{family_slug}/gaming/</span>')
    rec_lines.append(f'<span class="tree-plp">│   ├── /{family_slug}/baratos/</span>')
    rec_lines.append(f'<span class="tree-plp">│   └── /{family_slug}/{{marca}}/</span> <span class="tree-comment">← Si hay volumen</span>')
    rec_lines.append('│')
    
    # PDPs
    rec_lines.append('<span class="tree-pdp">├── 📦 Productos (PDPs en raíz)</span>')
    rec_lines.append(f'<span class="tree-pdp">│   ├── /{family_slug}-{{marca}}-{{modelo}}-{{specs}}</span>')
    rec_lines.append(f'<span class="tree-comment">│   │   └─ canonical → PLP padre</span>')
    rec_lines.append(f'<span class="tree-comment">│   │   └─ Target: long-tail específico</span>')
    rec_lines.append(f'<span class="tree-pdp">│   └── (estructura actual OK si no canibaliza)</span>')
    rec_lines.append('│')
    
    # Blog
    rec_lines.append(f'<span class="tree-blog">└── 📝 /blog/{family_slug}/</span> <span class="tree-comment">← Contenido informacional</span>')
    rec_lines.append(f'<span class="tree-blog">    ├── /blog/mejores-{family_slug}-2025/</span>')
    rec_lines.append(f'<span class="tree-blog">    ├── /blog/guia-compra-{family_slug}/</span>')
    rec_lines.append(f'<span class="tree-blog">    └── /blog/comparativa-{family_slug}/</span>')
    rec_lines.append(f'<span class="tree-comment">        └─ Enlaza a PLP, NO compite transaccional</span>')
    
    recommended_tree = '\n'.join(rec_lines)
    
    return current_tree, recommended_tree


# ==================== HELPERS ====================
def badge(text: str, variant: str) -> str:
    return f'<span class="badge badge-{variant}">{text}</span>'

def score_color(score: float) -> str:
    if score >= 70:
        return "score-good"
    elif score >= 40:
        return "score-warning"
    return "score-bad"

def health_bar_color(score: float) -> str:
    if score >= 70:
        return "#16a34a"
    elif score >= 40:
        return "#d97706"
    return "#dc2626"


# ==================== MAIN ====================
def main():
    st.title("🎯 SEO Cannibalization Analyzer v3.1")
    st.caption("Análisis por familia · Arquitectura óptima · Recomendaciones priorizadas")
    
    # ========== SIDEBAR ==========
    with st.sidebar:
        st.header("📁 Cargar datos")
        
        gsc_file = st.file_uploader(
            "CSV Search Console *", 
            type=['csv'], 
            key="gsc",
            help="Columnas: url, top_query, top_query_clicks, top_query_impressions, top_query_position"
        )
        
        cat_file = st.file_uploader(
            "CSV Categorías (opcional)", 
            type=['csv'], 
            key="cat",
            help="Archivo con slugs de categorías conocidas para mejorar clasificación"
        )
        
        adobe_file = st.file_uploader(
            "CSV Adobe Analytics (opcional)", 
            type=['csv'], 
            key="adobe",
            help="Para añadir datos de sesiones"
        )
        
        st.divider()
        
        st.header("⚙️ Configuración")
        min_clicks = st.slider("Mín. clics para analizar", 0, 100, 5)
        recovery_pct = st.slider(
            "% estimado de clics recuperables", 
            10, 80, 40,
            help="Estimación de qué porcentaje de clics dispersos se pueden recuperar consolidando"
        ) / 100
        
        st.divider()
        
        st.header("🔑 APIs (opcional)")
        anthropic_key = st.text_input("Anthropic API Key", type="password")
        openai_key = st.text_input("OpenAI API Key", type="password")
    
    # ========== PROCESAR DATOS ==========
    df = None
    categories_df = None
    known_plps = tuple()
    
    if gsc_file:
        content = gsc_file.read()
        gsc_file.seek(0)
        df = read_csv_flexible(content, "Search Console")
        
        if cat_file:
            cat_content = cat_file.read()
            cat_file.seek(0)
            categories_df = read_csv_flexible(cat_content, "Categorías")
            if categories_df is not None:
                known_plps = tuple(get_known_plp_slugs(categories_df))
                st.sidebar.success(f"✅ {len(known_plps)} categorías cargadas")
        
        if df is not None:
            # Clasificar URLs
            classifications = df['url'].apply(lambda x: classify_url_with_context(x, known_plps))
            df['url_type'] = classifications.apply(lambda x: x[0])
            df['family'] = classifications.apply(lambda x: x[1])
            df['depth'] = classifications.apply(lambda x: x[2])
            df['slug'] = df['url'].apply(extract_slug)
            
            # Asegurar columnas numéricas
            for col in ['top_query_clicks', 'top_query_impressions', 'top_query_position']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            st.session_state['df'] = df
            st.session_state['known_plps'] = known_plps
        
        # Adobe Analytics
        if adobe_file and df is not None:
            adobe_content = adobe_file.read()
            adobe_file.seek(0)
            adobe_df = read_csv_flexible(adobe_content, "Adobe", skip_comments=True)
            
            if adobe_df is not None:
                url_names = ['url', 'page', 'landing page', 'page url', 'pages']
                sessions_names = ['sessions', 'visits', 'pageviews', 'visitors']
                
                url_col = next((c for c in adobe_df.columns if c.lower().strip() in url_names), None)
                sessions_col = next((c for c in adobe_df.columns if c.lower().strip() in sessions_names), None)
                
                if url_col and sessions_col:
                    adobe_df = adobe_df.rename(columns={url_col: 'url', sessions_col: 'sessions'})
                    adobe_df['sessions'] = pd.to_numeric(
                        adobe_df['sessions'].astype(str).str.replace(',', '').str.replace(' ', ''),
                        errors='coerce'
                    ).fillna(0).astype(int)
                    
                    df = df.merge(adobe_df[['url', 'sessions']], on='url', how='left')
                    df['sessions'] = df['sessions'].fillna(0).astype(int)
                    st.session_state['df'] = df
                    st.sidebar.success(f"✅ Adobe: {len(adobe_df)} filas")
                else:
                    st.sidebar.warning(f"⚠️ Adobe: Columnas no reconocidas: {list(adobe_df.columns)[:5]}")
    
    # ========== TABS ==========
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard",
        "🔍 Canibalizaciones", 
        "🌳 Arquitectura",
        "💡 Recomendaciones"
    ])
    
    # ==================== TAB 1: DASHBOARD ====================
    with tab1:
        if df is None:
            st.info("👈 Sube tu archivo CSV de Search Console para comenzar")
            
            with st.expander("📋 Formato esperado del CSV"):
                st.markdown("""
                | Columna | Descripción |
                |---------|-------------|
                | `url` | URL completa |
                | `top_query` | Query principal de la URL |
                | `top_query_clicks` | Clics de esa query |
                | `top_query_impressions` | Impresiones |
                | `top_query_position` | Posición media |
                """)
            
            with st.expander("💡 ¿Por qué subir el archivo de categorías?"):
                st.markdown("""
                El archivo de categorías ayuda a **clasificar correctamente** las URLs:
                - Identifica qué slugs son PLPs conocidas
                - Mejora la detección de blogs vs productos
                - Reduce falsos positivos en la clasificación
                """)
        else:
            st.header("📊 Dashboard por Familia")
            
            # Métricas globales
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("URLs totales", f"{len(df):,}")
            with col2:
                st.metric("Clics totales", f"{int(df['top_query_clicks'].sum()):,}")
            with col3:
                st.metric("Familias", df['family'].nunique())
            with col4:
                type_dist = df['url_type'].value_counts()
                st.metric("PLPs / PDPs / Blog", f"{type_dist.get('PLP', 0)} / {type_dist.get('PDP', 0)} / {type_dist.get('BLOG', 0)}")
            
            st.divider()
            
            # Selector de familia
            families = df['family'].value_counts()
            family_options = [f"{f} ({c} URLs)" for f, c in families.head(50).items()]
            
            col1, col2 = st.columns([3, 1])
            with col1:
                selected = st.selectbox("Selecciona familia a analizar", family_options)
                selected_family = selected.split(' (')[0] if selected else None
            with col2:
                custom = st.text_input("O buscar manualmente:")
                if custom.strip():
                    selected_family = custom.strip().lower()
            
            if selected_family:
                st.session_state['selected_family'] = selected_family
                
                # Análisis de salud
                health = analyze_family_health(df, selected_family)
                
                if health is None:
                    st.warning(f"No se encontraron datos para la familia '{selected_family}'")
                else:
                    st.subheader(f"📈 Salud SEO: {selected_family.title()}")
                    
                    # Score y métricas
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        color = score_color(health.health_score)
                        st.markdown(f"""
                        <div class="score-card">
                            <div class="score-value {color}">{health.health_score:.0f}</div>
                            <div class="score-label">Health Score</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("URLs", health.total_urls)
                        st.caption(f"PLP: {health.plp_count} · PDP: {health.pdp_count} · Blog: {health.blog_count}")
                    
                    with col3:
                        st.metric("Clics", f"{health.total_clicks:,}")
                    
                    with col4:
                        st.metric("Queries canibalizadas", health.cannibalized_queries)
                    
                    with col5:
                        st.metric("Clics afectados", f"{health.cannibalized_clicks:,}")
                    
                    # Health bar
                    st.markdown(f"""
                    <div class="health-bar">
                        <div class="health-bar-fill" style="width: {health.health_score}%; background: {health_bar_color(health.health_score)};"></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Issues
                    if health.issues:
                        st.subheader("⚠️ Problemas detectados")
                        for issue in health.issues:
                            st.markdown(f"- {issue}")
                    else:
                        st.success("✅ No se detectaron problemas críticos en esta familia")
                    
                    # Gráficos
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        type_data = {'Tipo': ['PLP', 'PDP', 'BLOG'], 
                                    'URLs': [health.plp_count, health.pdp_count, health.blog_count]}
                        fig = px.pie(type_data, values='URLs', names='Tipo',
                                   color_discrete_sequence=['#0284c7', '#7c3aed', '#059669'])
                        fig.update_layout(height=280, margin=dict(t=30, b=10))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Top queries
                        family_lower = selected_family.lower()
                        mask = df['family'].str.lower() == family_lower
                        family_df = df[mask]
                        top_q = family_df.groupby('top_query')['top_query_clicks'].sum().sort_values(ascending=True).tail(8)
                        
                        fig = px.bar(x=top_q.values, y=top_q.index, orientation='h',
                                   color_discrete_sequence=['#0284c7'])
                        fig.update_layout(height=280, margin=dict(t=30, b=10), yaxis_title="", xaxis_title="Clics")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # PLP Principal
                    st.markdown(f"""
                    <div class="card card-info">
                        <strong>🏠 PLP Principal:</strong> <span class="url-slug">/{health.main_plp}</span>
                    </div>
                    """, unsafe_allow_html=True)
    
    # ==================== TAB 2: CANIBALIZACIONES ====================
    with tab2:
        if df is None:
            st.info("👈 Carga datos en la pestaña Dashboard primero")
        elif 'selected_family' not in st.session_state:
            st.info("👈 Selecciona una familia en el Dashboard primero")
        else:
            family = st.session_state['selected_family']
            st.header(f"🔍 Canibalizaciones: {family.title()}")
            
            cannibalizations = detect_cannibalizations(df, family, min_clicks, recovery_pct)
            st.session_state['cannibalizations'] = cannibalizations
            
            if not cannibalizations:
                st.success("✅ No se detectaron canibalizaciones significativas")
            else:
                # Resumen
                total_recoverable = sum(c.recoverable_clicks for c in cannibalizations)
                critical = len([c for c in cannibalizations if c.severity == Severity.CRITICAL])
                high = len([c for c in cannibalizations if c.severity == Severity.HIGH])
                medium = len([c for c in cannibalizations if c.severity == Severity.MEDIUM])
                low = len([c for c in cannibalizations if c.severity == Severity.LOW])
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("🔴 Críticas", critical)
                with col2:
                    st.metric("🟠 Altas", high)
                with col3:
                    st.metric("🟡 Medias", medium)
                with col4:
                    st.metric("🟢 Bajas", low)
                with col5:
                    st.metric("💰 Recuperables", f"{total_recoverable:,}")
                
                st.divider()
                
                # Filtro
                severity_filter = st.radio(
                    "Filtrar severidad:",
                    ["Todas", "Críticas", "Altas", "Medias", "Bajas"],
                    horizontal=True
                )
                
                sev_map = {"Críticas": Severity.CRITICAL, "Altas": Severity.HIGH, 
                          "Medias": Severity.MEDIUM, "Bajas": Severity.LOW}
                
                filtered = cannibalizations if severity_filter == "Todas" else [
                    c for c in cannibalizations if c.severity == sev_map[severity_filter]
                ]
                
                st.caption(f"Mostrando {len(filtered)} de {len(cannibalizations)}")
                
                # Lista
                for cannib in filtered[:30]:
                    sev_emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}[cannib.severity.value]
                    
                    with st.expander(f"{sev_emoji} `{cannib.query}` — {len(cannib.urls)} URLs · {cannib.total_clicks:,} clics"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Tipo", cannib.issue_type)
                        with col2:
                            st.metric("Concentración", f"{cannib.click_concentration:.0f}%")
                        with col3:
                            st.metric("Spread pos.", f"{cannib.position_spread:.1f}")
                        with col4:
                            st.metric("Recuperables", f"{cannib.recoverable_clicks:,}")
                        
                        st.markdown(f"""
                        <div class="card card-info">
                            <strong>💡 Recomendación:</strong> {cannib.recommendation}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # URL Principal
                        main = cannib.main_url
                        st.markdown(f"""
                        <div class="card card-success">
                            {badge("PRINCIPAL", "low")} {badge(main['type'], main['type'].lower())}
                            <div class="url-slug" style="margin: 8px 0;">
                                <a href="{main['url']}" target="_blank">/{main['slug']}</a>
                            </div>
                            <div class="query-tag">{main['top_query'][:50]}</div>
                            <div class="metrics-inline">
                                <span>Clics: <strong>{main['clicks']:,}</strong></span>
                                <span>Impr: <strong>{main['impressions']:,}</strong></span>
                                <span>Pos: <strong>{main['position']:.1f}</strong></span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Competidoras
                        st.markdown("**Compitiendo:**")
                        for comp in cannib.competing_urls:
                            st.markdown(f"""
                            <div class="card card-danger">
                                {badge(comp['type'], comp['type'].lower())}
                                <div class="url-slug" style="margin: 8px 0;">
                                    <a href="{comp['url']}" target="_blank">/{comp['slug']}</a>
                                </div>
                                <div class="query-tag">{comp['top_query'][:50]}</div>
                                <div class="metrics-inline">
                                    <span>Clics: <strong>{comp['clicks']:,}</strong></span>
                                    <span>Impr: <strong>{comp['impressions']:,}</strong></span>
                                    <span>Pos: <strong>{comp['position']:.1f}</strong></span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
    
    # ==================== TAB 3: ARQUITECTURA ====================
    with tab3:
        if df is None:
            st.info("👈 Carga datos en la pestaña Dashboard primero")
        elif 'selected_family' not in st.session_state:
            st.info("👈 Selecciona una familia en el Dashboard primero")
        else:
            family = st.session_state['selected_family']
            st.header(f"🌳 Arquitectura: {family.title()}")
            
            cannibalizations = st.session_state.get('cannibalizations', [])
            current_tree, recommended_tree = generate_architecture_tree(df, family, cannibalizations)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📁 Estructura Actual")
                st.markdown(f'<div class="tree-container">{current_tree}</div>', unsafe_allow_html=True)
            
            with col2:
                st.subheader("✅ Estructura Recomendada")
                st.markdown(f'<div class="tree-container">{recommended_tree}</div>', unsafe_allow_html=True)
            
            # Reglas
            st.divider()
            st.subheader("📐 Reglas de Arquitectura SEO")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="card card-success">
                    <strong>✅ Buenas prácticas</strong>
                    <ul style="margin: 0.5rem 0; padding-left: 20px;">
                        <li><strong>PLP principal</strong> en /{familia}/ → queries transaccionales</li>
                        <li><strong>Subcategorías</strong> con / para segmentar</li>
                        <li><strong>PDPs</strong> con slug descriptivo en raíz + canonical a PLP</li>
                        <li><strong>Blog</strong> en /blog/{familia}/ → queries informacionales</li>
                        <li>Máximo 3 clics desde home</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="card card-danger">
                    <strong>❌ Anti-patrones</strong>
                    <ul style="margin: 0.5rem 0; padding-left: 20px;">
                        <li>Blog en raíz compitiendo con PLP comercial</li>
                        <li>Múltiples PLPs para mismo intent</li>
                        <li>PDPs rankeando por queries genéricas sin canonical</li>
                        <li>Profundidad > 4 niveles</li>
                        <li>Páginas huérfanas sin enlaces internos</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    # ==================== TAB 4: RECOMENDACIONES ====================
    with tab4:
        if df is None:
            st.info("👈 Carga datos en la pestaña Dashboard primero")
        elif 'selected_family' not in st.session_state:
            st.info("👈 Selecciona una familia en el Dashboard primero")
        elif 'cannibalizations' not in st.session_state or not st.session_state['cannibalizations']:
            st.info("No hay canibalizaciones que recomendar resolver")
        else:
            family = st.session_state['selected_family']
            cannibalizations = st.session_state['cannibalizations']
            
            st.header(f"💡 Recomendaciones: {family.title()}")
            
            st.markdown("""
            <div class="card card-warning">
                ⚠️ <strong>Disclaimer:</strong> Validar todas las acciones con el equipo SEO antes de implementar.
            </div>
            """, unsafe_allow_html=True)
            
            # Total recuperable
            total_recoverable = sum(c.recoverable_clicks for c in cannibalizations)
            st.metric("💰 Total clics potencialmente recuperables", f"{total_recoverable:,}")
            st.caption(f"Basado en {int(recovery_pct*100)}% de recuperación estimada (configurable en sidebar)")
            
            st.divider()
            
            # Separar acciones
            hard_actions = []
            soft_actions = []
            
            for c in cannibalizations:
                for comp in c.competing_urls:
                    item = {
                        'query': c.query,
                        'url': comp['url'],
                        'slug': comp['slug'],
                        'type': comp['type'],
                        'clicks': comp['clicks'],
                        'main_slug': c.main_url['slug'],
                        'issue': c.issue_type,
                        'severity': c.severity.value
                    }
                    
                    if c.issue_type in ["Blog vs PLP", "PLPs duplicadas", "Blogs duplicados"]:
                        hard_actions.append(item)
                    else:
                        soft_actions.append(item)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"🔴 Acciones Hard ({len(hard_actions)})")
                st.caption("301 redirects, cambios estructurales")
                
                for item in sorted(hard_actions, key=lambda x: x['clicks'], reverse=True)[:15]:
                    st.markdown(f"""
                    <div class="card card-danger">
                        {badge(item['type'], item['type'].lower())} {badge(item['severity'], item['severity'].lower())}
                        <div style="font-weight: 600; margin: 8px 0;">301 → /{item['main_slug'][:35]}...</div>
                        <div class="url-slug">/{item['slug'][:45]}...</div>
                        <div class="query-tag" style="margin-top: 8px;">{item['query'][:35]}</div>
                        <div style="font-size: 12px; margin-top: 4px;">Clics: <strong>{item['clicks']}</strong></div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.subheader(f"🟡 Acciones Soft ({len(soft_actions)})")
                st.caption("Canonical, diferenciación, optimización")
                
                for item in sorted(soft_actions, key=lambda x: x['clicks'], reverse=True)[:15]:
                    action = "Canonical" if item['issue'] == "PDP vs PLP" else "Diferenciar"
                    st.markdown(f"""
                    <div class="card">
                        {badge(item['type'], item['type'].lower())}
                        <div style="font-weight: 600; margin: 8px 0;">{action}</div>
                        <div class="url-slug">/{item['slug'][:45]}...</div>
                        <div class="query-tag" style="margin-top: 8px;">{item['query'][:35]}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Exportar
            st.divider()
            if st.button("📥 Exportar CSV", use_container_width=True):
                export_data = []
                for c in cannibalizations:
                    for comp in c.competing_urls:
                        export_data.append({
                            'familia': family,
                            'query': c.query,
                            'url_competidora': comp['url'],
                            'slug': comp['slug'],
                            'tipo': comp['type'],
                            'clics': comp['clicks'],
                            'url_principal': c.main_url['url'],
                            'issue_type': c.issue_type,
                            'severidad': c.severity.value,
                            'recomendacion': c.recommendation,
                            'clics_recuperables': c.recoverable_clicks
                        })
                
                csv = pd.DataFrame(export_data).to_csv(index=False)
                st.download_button("💾 Descargar", csv, f"recomendaciones_{family}.csv", "text/csv")


if __name__ == "__main__":
    main()
