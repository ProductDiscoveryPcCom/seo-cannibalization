"""
SEO Cannibalization Analyzer v3.2
Herramienta profesional de análisis de canibalizaciones SEO

CAMBIOS v3.2:
- Añadido st.status() y st.progress() para feedback visual
- Procesamiento con indicadores de progreso
- Mejor manejo de archivos grandes
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
import time

# ==================== CONFIGURACIÓN ====================
st.set_page_config(
    page_title="SEO Cannibalization Analyzer v3",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; max-width: 1400px; }
    
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
    
    .card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }
    .card-danger { border-left: 4px solid #dc2626; background: #fef2f2; }
    .card-warning { border-left: 4px solid #d97706; background: #fffbeb; }
    .card-success { border-left: 4px solid #16a34a; background: #f0fdf4; }
    .card-info { border-left: 4px solid #0284c7; background: #eff6ff; }
    
    .badge { display: inline-block; padding: 3px 10px; border-radius: 4px; font-size: 11px; font-weight: 600; }
    .badge-plp { background: #0284c7; color: white; }
    .badge-pdp { background: #7c3aed; color: white; }
    .badge-blog { background: #059669; color: white; }
    .badge-critical { background: #7f1d1d; color: white; }
    .badge-high { background: #dc2626; color: white; }
    .badge-medium { background: #d97706; color: white; }
    .badge-low { background: #16a34a; color: white; }
    
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
    
    .metrics-inline {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-top: 0.5rem;
        font-size: 12px;
        color: #64748b;
    }
    .metrics-inline strong { color: #0f172a; }
    
    .health-bar {
        height: 8px;
        background: #e2e8f0;
        border-radius: 4px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .health-bar-fill { height: 100%; border-radius: 4px; }
    
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


# ==================== FUNCIONES DE UTILIDAD ====================
def extract_slug(url: str) -> str:
    try:
        parsed = urlparse(url)
        return parsed.path.strip('/') or '/'
    except:
        return str(url)

def get_known_plp_slugs(categories_df: pd.DataFrame) -> Set[str]:
    """Extrae slugs conocidos de PLPs"""
    known_slugs = set()
    if categories_df is not None and not categories_df.empty:
        slug_col = None
        for col in categories_df.columns:
            if col.lower().strip() in ['slug', 'url', 'categoria', 'category']:
                slug_col = col
                break
        
        if slug_col:
            for val in categories_df[slug_col].dropna():
                if 'http' in str(val):
                    slug = extract_slug(str(val))
                else:
                    slug = str(val).strip('/').lower()
                if slug:
                    known_slugs.add(slug)
    return known_slugs

def classify_url_fast(url: str, known_plps: Set[str]) -> Tuple[str, str, int]:
    """Clasificación rápida de URL"""
    try:
        parsed = urlparse(url)
        path = parsed.path.lower().strip('/')
        
        if not path:
            return 'PLP', 'home', 0
        
        segments = [s for s in path.split('/') if s]
        first_segment = segments[0] if segments else ''
        
        # 1. Verificar PLPs conocidas
        if path in known_plps or first_segment in known_plps:
            family = first_segment.replace('-', ' ')
            return 'PLP', family, len(segments)
        
        # 2. Detectar BLOG
        blog_patterns = ['blog', 'noticias', 'guia', 'guias', 'magazine', 'revista', 'articulo']
        if any(p in path for p in blog_patterns):
            for seg in segments:
                if seg not in blog_patterns:
                    return 'BLOG', seg.replace('-', ' '), len(segments)
            return 'BLOG', 'general', len(segments)
        
        # 3. Un solo segmento
        if len(segments) == 1:
            slug = segments[0]
            if slug.count('-') >= 5:
                family = slug.split('-')[0]
                return 'PDP', family, 1
            else:
                return 'PLP', slug.replace('-', ' '), 1
        
        # 4. Múltiples segmentos
        family = first_segment.replace('-', ' ')
        last_segment = segments[-1]
        
        if last_segment.count('-') >= 5:
            return 'PDP', family, len(segments)
        
        return 'PLP', family, len(segments)
        
    except:
        return 'OTHER', 'unknown', 0


def read_csv_simple(file, skip_comments: bool = False) -> pd.DataFrame:
    """Lee CSV de forma simple y robusta"""
    try:
        content = file.read()
        file.seek(0)
        
        # Decodificar
        try:
            text = content.decode('utf-8')
        except:
            try:
                text = content.decode('latin-1')
            except:
                text = content.decode('cp1252')
        
        lines = text.split('\n')
        
        # Saltar comentarios
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
        
        df = pd.read_csv(
            StringIO(text), 
            sep=sep, 
            skiprows=skip_rows,
            on_bad_lines='skip',
            low_memory=False
        )
        df.columns = df.columns.str.strip()
        return df
        
    except Exception as e:
        st.error(f"Error leyendo archivo: {str(e)}")
        return None


# ==================== PROCESAMIENTO CON PROGRESO ====================
def process_data_with_progress(gsc_file, cat_file, adobe_file):
    """Procesa los datos mostrando progreso"""
    
    df = None
    categories_df = None
    known_plps = set()
    
    with st.status("🔄 Procesando datos...", expanded=True) as status:
        
        # Paso 1: Leer CSV principal
        st.write("📖 Leyendo archivo de Search Console...")
        df = read_csv_simple(gsc_file)
        
        if df is None:
            status.update(label="❌ Error al leer archivo", state="error")
            return None, set()
        
        st.write(f"   ✓ {len(df):,} filas cargadas")
        
        # Paso 2: Cargar categorías si existe
        if cat_file:
            st.write("📂 Leyendo archivo de categorías...")
            categories_df = read_csv_simple(cat_file)
            if categories_df is not None:
                known_plps = get_known_plp_slugs(categories_df)
                st.write(f"   ✓ {len(known_plps)} categorías conocidas")
        
        # Paso 3: Clasificar URLs (con progress bar)
        st.write("🏷️ Clasificando URLs...")
        
        total_rows = len(df)
        progress_bar = st.progress(0)
        
        # Procesar en chunks para mostrar progreso
        chunk_size = max(1000, total_rows // 100)
        url_types = []
        families = []
        depths = []
        slugs = []
        
        for i in range(0, total_rows, chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            
            for url in chunk['url']:
                slug = extract_slug(url)
                url_type, family, depth = classify_url_fast(url, known_plps)
                url_types.append(url_type)
                families.append(family)
                depths.append(depth)
                slugs.append(slug)
            
            progress = min((i + chunk_size) / total_rows, 1.0)
            progress_bar.progress(progress)
        
        progress_bar.progress(1.0)
        
        df['url_type'] = url_types
        df['family'] = families
        df['depth'] = depths
        df['slug'] = slugs
        
        # Paso 4: Convertir columnas numéricas
        st.write("🔢 Procesando métricas...")
        for col in ['top_query_clicks', 'top_query_impressions', 'top_query_position']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Paso 5: Adobe Analytics (opcional)
        if adobe_file:
            st.write("📊 Procesando Adobe Analytics...")
            adobe_df = read_csv_simple(adobe_file, skip_comments=True)
            
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
                    st.write(f"   ✓ Adobe Analytics fusionado: {len(adobe_df):,} filas")
                else:
                    st.write(f"   ⚠️ Columnas Adobe no reconocidas")
        
        # Resumen final
        type_counts = df['url_type'].value_counts()
        st.write(f"📈 Resumen: {type_counts.get('PLP', 0)} PLPs, {type_counts.get('PDP', 0)} PDPs, {type_counts.get('BLOG', 0)} Blogs")
        
        status.update(label="✅ Datos procesados correctamente", state="complete")
    
    return df, known_plps


# ==================== ANÁLISIS ====================
def analyze_family_health(df: pd.DataFrame, family: str) -> Optional[FamilyHealth]:
    """Analiza la salud SEO de una familia"""
    
    family_lower = family.lower()
    family_slug = family_lower.replace(' ', '-')
    
    mask = (
        (df['family'].str.lower() == family_lower) |
        (df['slug'].str.lower().str.startswith(family_slug))
    )
    family_df = df[mask].copy()
    
    if len(family_df) == 0:
        return None
    
    plp_count = len(family_df[family_df['url_type'] == 'PLP'])
    pdp_count = len(family_df[family_df['url_type'] == 'PDP'])
    blog_count = len(family_df[family_df['url_type'] == 'BLOG'])
    
    total_clicks = int(family_df['top_query_clicks'].sum())
    total_impressions = int(family_df['top_query_impressions'].sum())
    
    query_counts = family_df.groupby('top_query').size()
    cannibalized_queries = int((query_counts > 1).sum())
    
    cannib_queries = query_counts[query_counts > 1].index.tolist()
    cannib_df = family_df[family_df['top_query'].isin(cannib_queries)]
    cannibalized_clicks = int(cannib_df['top_query_clicks'].sum())
    
    plps = family_df[family_df['url_type'] == 'PLP'].sort_values('top_query_clicks', ascending=False)
    main_plp = plps.iloc[0]['slug'] if len(plps) > 0 else "No hay PLP"
    
    issues = []
    if plp_count == 0:
        issues.append("❌ No hay PLP principal")
    if plp_count > 3:
        issues.append(f"⚠️ Demasiadas PLPs ({plp_count})")
    if cannibalized_queries > 5:
        issues.append(f"🔴 {cannibalized_queries} queries canibalizadas")
    if total_clicks > 0 and cannibalized_clicks > total_clicks * 0.3:
        pct = cannibalized_clicks / total_clicks * 100
        issues.append(f"⚠️ {pct:.0f}% tráfico afectado")
    
    health_score = 100
    if plp_count == 0:
        health_score -= 30
    if cannibalized_queries > 0:
        health_score -= min(30, cannibalized_queries * 2)
    if total_clicks > 0 and cannibalized_clicks > total_clicks * 0.3:
        health_score -= 20
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
    """Detecta canibalizaciones en una familia"""
    
    family_lower = family.lower()
    family_slug = family_lower.replace(' ', '-')
    
    mask = (
        (df['family'].str.lower() == family_lower) |
        (df['slug'].str.lower().str.startswith(family_slug))
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
        
        types_involved = set(u['type'] for u in urls_data)
        
        if 'BLOG' in types_involved and 'PLP' in types_involved:
            issue_type = "Blog vs PLP"
            recommendation = "Redirigir blog a /blog/ o añadir canonical"
        elif 'PDP' in types_involved and 'PLP' in types_involved:
            issue_type = "PDP vs PLP"
            recommendation = "Añadir canonical en PDP hacia PLP"
        elif types_involved == {'PLP'}:
            issue_type = "PLPs duplicadas"
            recommendation = "Consolidar en una PLP y 301 el resto"
        elif types_involved == {'PDP'}:
            issue_type = "PDPs compitiendo"
            recommendation = "Diferenciar contenido"
        elif types_involved == {'BLOG'}:
            issue_type = "Blogs duplicados"
            recommendation = "Fusionar artículos"
        else:
            issue_type = "Mixto"
            recommendation = "Analizar caso por caso"
        
        url_count = len(urls_data)
        if url_count >= 4 or (issue_type == "Blog vs PLP" and click_concentration < 60):
            severity = Severity.CRITICAL
        elif url_count >= 3 or issue_type in ["Blog vs PLP", "PLPs duplicadas"]:
            severity = Severity.HIGH
        elif click_concentration < 70:
            severity = Severity.MEDIUM
        else:
            severity = Severity.LOW
        
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
    
    severity_order = {Severity.CRITICAL: 4, Severity.HIGH: 3, Severity.MEDIUM: 2, Severity.LOW: 1}
    cannibalizations.sort(key=lambda x: (severity_order[x.severity], x.recoverable_clicks), reverse=True)
    
    return cannibalizations


# ==================== ÁRBOL ====================
def generate_architecture_tree(df: pd.DataFrame, family: str, cannibalizations: List[Cannibalization]) -> Tuple[str, str]:
    """Genera árboles de arquitectura"""
    
    family_lower = family.lower()
    family_slug = family.replace(' ', '-')
    
    mask = (
        (df['family'].str.lower() == family_lower) |
        (df['slug'].str.lower().str.startswith(family_slug))
    )
    family_df = df[mask].copy()
    
    if len(family_df) == 0:
        return "No hay datos", "No hay datos"
    
    problem_queries = {c.query for c in cannibalizations}
    
    plps = family_df[family_df['url_type'] == 'PLP'].sort_values('top_query_clicks', ascending=False)
    pdps = family_df[family_df['url_type'] == 'PDP'].sort_values('top_query_clicks', ascending=False)
    blogs = family_df[family_df['url_type'] == 'BLOG'].sort_values('top_query_clicks', ascending=False)
    
    # ÁRBOL ACTUAL
    lines = []
    lines.append(f'<span class="tree-plp">📁 /{family_slug}/</span>')
    lines.append(f'<span class="tree-comment">   {len(family_df)} URLs</span>')
    lines.append('')
    
    if len(plps) > 0:
        lines.append(f'<span class="tree-plp">├── 📂 PLPs ({len(plps)})</span>')
        for i, (_, row) in enumerate(plps.head(5).iterrows()):
            slug_short = str(row['slug'])[:35]
            has_problem = row['top_query'] in problem_queries
            prefix = '│   ├──' if i < min(len(plps)-1, 4) else '│   └──'
            if has_problem:
                lines.append(f'<span class="tree-problem">{prefix} ⚠️ /{slug_short}</span>')
            else:
                lines.append(f'<span class="tree-ok">{prefix} ✓ /{slug_short}</span>')
        if len(plps) > 5:
            lines.append(f'<span class="tree-comment">│   └── +{len(plps)-5} más</span>')
    
    lines.append('│')
    
    if len(pdps) > 0:
        problem_pdps = len(pdps[pdps['top_query'].isin(problem_queries)])
        lines.append(f'<span class="tree-pdp">├── 📦 PDPs ({len(pdps)})</span>')
        if problem_pdps > 0:
            lines.append(f'<span class="tree-problem">│   ├── ⚠️ {problem_pdps} con conflicto</span>')
        lines.append(f'<span class="tree-ok">│   └── ✓ {len(pdps)-problem_pdps} OK</span>')
    
    lines.append('│')
    
    if len(blogs) > 0:
        lines.append(f'<span class="tree-blog">└── 📝 Blog ({len(blogs)})</span>')
        for i, (_, row) in enumerate(blogs.head(3).iterrows()):
            slug_short = str(row['slug'])[:30]
            has_problem = row['top_query'] in problem_queries
            prefix = '    ├──' if i < min(len(blogs)-1, 2) else '    └──'
            if has_problem:
                lines.append(f'<span class="tree-problem">{prefix} ⚠️ /{slug_short}</span>')
            else:
                lines.append(f'<span class="tree-ok">{prefix} ✓ /{slug_short}</span>')
    
    current_tree = '\n'.join(lines)
    
    # ÁRBOL RECOMENDADO
    rec = []
    rec.append(f'<span class="tree-plp">📁 /{family_slug}/</span> <span class="tree-comment">← PLP Principal</span>')
    rec.append('')
    rec.append('<span class="tree-plp">├── 📂 Subcategorías</span>')
    rec.append(f'<span class="tree-plp">│   ├── /{family_slug}/gaming/</span>')
    rec.append(f'<span class="tree-plp">│   ├── /{family_slug}/baratos/</span>')
    rec.append(f'<span class="tree-plp">│   └── /{family_slug}/{{marca}}/</span>')
    rec.append('│')
    rec.append('<span class="tree-pdp">├── 📦 PDPs (en raíz)</span>')
    rec.append(f'<span class="tree-pdp">│   └── /{family_slug}-marca-modelo</span>')
    rec.append(f'<span class="tree-comment">│       └─ canonical → PLP</span>')
    rec.append('│')
    rec.append(f'<span class="tree-blog">└── 📝 /blog/{family_slug}/</span>')
    rec.append(f'<span class="tree-blog">    └── /blog/mejores-{family_slug}/</span>')
    rec.append(f'<span class="tree-comment">        └─ NO compite con PLP</span>')
    
    recommended_tree = '\n'.join(rec)
    
    return current_tree, recommended_tree


# ==================== HELPERS ====================
def badge(text: str, variant: str) -> str:
    return f'<span class="badge badge-{variant}">{text}</span>'

def score_color(score: float) -> str:
    if score >= 70: return "score-good"
    elif score >= 40: return "score-warning"
    return "score-bad"

def health_bar_color(score: float) -> str:
    if score >= 70: return "#16a34a"
    elif score >= 40: return "#d97706"
    return "#dc2626"


# ==================== MAIN ====================
def main():
    st.title("🎯 SEO Cannibalization Analyzer v3.2")
    st.caption("Análisis por familia · Arquitectura óptima · Recomendaciones priorizadas")
    
    # SIDEBAR
    with st.sidebar:
        st.header("📁 Cargar datos")
        
        gsc_file = st.file_uploader("CSV Search Console *", type=['csv'], key="gsc")
        cat_file = st.file_uploader("CSV Categorías (opcional)", type=['csv'], key="cat")
        adobe_file = st.file_uploader("CSV Adobe Analytics (opcional)", type=['csv'], key="adobe")
        
        st.divider()
        
        st.header("⚙️ Configuración")
        min_clicks = st.slider("Mín. clics", 0, 100, 5)
        recovery_pct = st.slider("% recuperación estimado", 10, 80, 40) / 100
        
        st.divider()
        
        st.header("🔑 APIs (opcional)")
        anthropic_key = st.text_input("Anthropic API Key", type="password")
    
    # PROCESAR DATOS
    df = None
    
    if gsc_file is not None:
        # Verificar si ya procesamos este archivo
        file_id = f"{gsc_file.name}_{gsc_file.size}"
        
        if 'processed_file' not in st.session_state or st.session_state.get('processed_file') != file_id:
            # Procesar con indicadores de progreso
            df, known_plps = process_data_with_progress(gsc_file, cat_file, adobe_file)
            
            if df is not None:
                st.session_state['df'] = df
                st.session_state['known_plps'] = known_plps
                st.session_state['processed_file'] = file_id
        else:
            df = st.session_state.get('df')
    
    # TABS
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard",
        "🔍 Canibalizaciones", 
        "🌳 Arquitectura",
        "💡 Recomendaciones"
    ])
    
    # TAB 1: DASHBOARD
    with tab1:
        if df is None:
            st.info("👈 Sube tu archivo CSV de Search Console para comenzar")
            
            with st.expander("📋 Formato esperado"):
                st.markdown("""
                | Columna | Descripción |
                |---------|-------------|
                | `url` | URL completa |
                | `top_query` | Query principal |
                | `top_query_clicks` | Clics |
                | `top_query_impressions` | Impresiones |
                | `top_query_position` | Posición |
                """)
        else:
            st.header("📊 Dashboard por Familia")
            
            # Métricas globales
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("URLs", f"{len(df):,}")
            with col2:
                st.metric("Clics", f"{int(df['top_query_clicks'].sum()):,}")
            with col3:
                st.metric("Familias", df['family'].nunique())
            with col4:
                type_dist = df['url_type'].value_counts()
                st.metric("PLP/PDP/Blog", f"{type_dist.get('PLP',0)}/{type_dist.get('PDP',0)}/{type_dist.get('BLOG',0)}")
            
            st.divider()
            
            # Selector de familia
            families = df['family'].value_counts()
            family_options = [f"{f} ({c} URLs)" for f, c in families.head(50).items()]
            
            col1, col2 = st.columns([3, 1])
            with col1:
                selected = st.selectbox("Selecciona familia", family_options)
                selected_family = selected.split(' (')[0] if selected else None
            with col2:
                custom = st.text_input("O buscar:")
                if custom.strip():
                    selected_family = custom.strip().lower()
            
            if selected_family:
                st.session_state['selected_family'] = selected_family
                
                with st.spinner(f"Analizando {selected_family}..."):
                    health = analyze_family_health(df, selected_family)
                
                if health is None:
                    st.warning(f"No se encontraron datos para '{selected_family}'")
                else:
                    st.subheader(f"📈 Salud SEO: {selected_family.title()}")
                    
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
                        st.caption(f"PLP:{health.plp_count} PDP:{health.pdp_count} Blog:{health.blog_count}")
                    
                    with col3:
                        st.metric("Clics", f"{health.total_clicks:,}")
                    
                    with col4:
                        st.metric("Q. canibalizadas", health.cannibalized_queries)
                    
                    with col5:
                        st.metric("Clics afectados", f"{health.cannibalized_clicks:,}")
                    
                    st.markdown(f"""
                    <div class="health-bar">
                        <div class="health-bar-fill" style="width:{health.health_score}%;background:{health_bar_color(health.health_score)};"></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if health.issues:
                        st.subheader("⚠️ Problemas")
                        for issue in health.issues:
                            st.markdown(f"- {issue}")
                    
                    # Gráficos
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.pie(
                            values=[health.plp_count, health.pdp_count, health.blog_count],
                            names=['PLP', 'PDP', 'Blog'],
                            color_discrete_sequence=['#0284c7', '#7c3aed', '#059669']
                        )
                        fig.update_layout(height=250, margin=dict(t=20,b=20))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        family_df = df[df['family'].str.lower() == selected_family.lower()]
                        top_q = family_df.groupby('top_query')['top_query_clicks'].sum().sort_values(ascending=True).tail(8)
                        fig = px.bar(x=top_q.values, y=top_q.index, orientation='h', color_discrete_sequence=['#0284c7'])
                        fig.update_layout(height=250, margin=dict(t=20,b=20), yaxis_title="")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown(f"""
                    <div class="card card-info">
                        <strong>🏠 PLP Principal:</strong> <code>/{health.main_plp}</code>
                    </div>
                    """, unsafe_allow_html=True)
    
    # TAB 2: CANIBALIZACIONES
    with tab2:
        if df is None:
            st.info("👈 Carga datos primero")
        elif 'selected_family' not in st.session_state:
            st.info("👈 Selecciona una familia en Dashboard")
        else:
            family = st.session_state['selected_family']
            st.header(f"🔍 Canibalizaciones: {family.title()}")
            
            with st.spinner("Detectando canibalizaciones..."):
                cannibalizations = detect_cannibalizations(df, family, min_clicks, recovery_pct)
            st.session_state['cannibalizations'] = cannibalizations
            
            if not cannibalizations:
                st.success("✅ No se detectaron canibalizaciones")
            else:
                total_recoverable = sum(c.recoverable_clicks for c in cannibalizations)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                critical = len([c for c in cannibalizations if c.severity == Severity.CRITICAL])
                high = len([c for c in cannibalizations if c.severity == Severity.HIGH])
                medium = len([c for c in cannibalizations if c.severity == Severity.MEDIUM])
                low = len([c for c in cannibalizations if c.severity == Severity.LOW])
                
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
                
                severity_filter = st.radio("Filtrar:", ["Todas", "Críticas", "Altas", "Medias", "Bajas"], horizontal=True)
                
                sev_map = {"Críticas": Severity.CRITICAL, "Altas": Severity.HIGH, "Medias": Severity.MEDIUM, "Bajas": Severity.LOW}
                filtered = cannibalizations if severity_filter == "Todas" else [c for c in cannibalizations if c.severity == sev_map[severity_filter]]
                
                st.caption(f"Mostrando {len(filtered)} de {len(cannibalizations)}")
                
                for cannib in filtered[:25]:
                    sev_emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}[cannib.severity.value]
                    
                    with st.expander(f"{sev_emoji} `{cannib.query}` — {len(cannib.urls)} URLs · {cannib.total_clicks:,} clics"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Tipo", cannib.issue_type)
                        with col2:
                            st.metric("Concentración", f"{cannib.click_concentration:.0f}%")
                        with col3:
                            st.metric("Spread", f"{cannib.position_spread:.1f}")
                        with col4:
                            st.metric("Recuperables", f"{cannib.recoverable_clicks:,}")
                        
                        st.markdown(f"""<div class="card card-info"><strong>💡</strong> {cannib.recommendation}</div>""", unsafe_allow_html=True)
                        
                        main = cannib.main_url
                        st.markdown(f"""
                        <div class="card card-success">
                            {badge("PRINCIPAL", "low")} {badge(main['type'], main['type'].lower())}
                            <div class="url-slug" style="margin:8px 0;"><a href="{main['url']}" target="_blank">/{main['slug']}</a></div>
                            <div class="query-tag">{main['top_query'][:50]}</div>
                            <div class="metrics-inline">
                                <span>Clics: <strong>{main['clicks']:,}</strong></span>
                                <span>Pos: <strong>{main['position']:.1f}</strong></span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for comp in cannib.competing_urls:
                            st.markdown(f"""
                            <div class="card card-danger">
                                {badge(comp['type'], comp['type'].lower())}
                                <div class="url-slug" style="margin:8px 0;"><a href="{comp['url']}" target="_blank">/{comp['slug']}</a></div>
                                <div class="query-tag">{comp['top_query'][:50]}</div>
                                <div class="metrics-inline">
                                    <span>Clics: <strong>{comp['clicks']:,}</strong></span>
                                    <span>Pos: <strong>{comp['position']:.1f}</strong></span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
    
    # TAB 3: ARQUITECTURA
    with tab3:
        if df is None:
            st.info("👈 Carga datos primero")
        elif 'selected_family' not in st.session_state:
            st.info("👈 Selecciona una familia en Dashboard")
        else:
            family = st.session_state['selected_family']
            st.header(f"🌳 Arquitectura: {family.title()}")
            
            cannibalizations = st.session_state.get('cannibalizations', [])
            current_tree, recommended_tree = generate_architecture_tree(df, family, cannibalizations)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📁 Actual")
                st.markdown(f'<div class="tree-container">{current_tree}</div>', unsafe_allow_html=True)
            with col2:
                st.subheader("✅ Recomendada")
                st.markdown(f'<div class="tree-container">{recommended_tree}</div>', unsafe_allow_html=True)
            
            st.divider()
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <div class="card card-success">
                    <strong>✅ Buenas prácticas</strong>
                    <ul><li>PLP principal en /{familia}/</li><li>PDPs con canonical a PLP</li><li>Blog en /blog/{familia}/</li></ul>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown("""
                <div class="card card-danger">
                    <strong>❌ Evitar</strong>
                    <ul><li>Blog en raíz compitiendo con PLP</li><li>Múltiples PLPs mismo intent</li><li>PDPs sin canonical</li></ul>
                </div>
                """, unsafe_allow_html=True)
    
    # TAB 4: RECOMENDACIONES
    with tab4:
        if df is None or 'selected_family' not in st.session_state:
            st.info("👈 Selecciona una familia primero")
        elif 'cannibalizations' not in st.session_state or not st.session_state['cannibalizations']:
            st.info("No hay canibalizaciones detectadas")
        else:
            family = st.session_state['selected_family']
            cannibalizations = st.session_state['cannibalizations']
            
            st.header(f"💡 Recomendaciones: {family.title()}")
            
            st.markdown("""<div class="card card-warning">⚠️ Validar con el equipo SEO antes de implementar</div>""", unsafe_allow_html=True)
            
            total_recoverable = sum(c.recoverable_clicks for c in cannibalizations)
            st.metric("💰 Clics recuperables estimados", f"{total_recoverable:,}")
            
            st.divider()
            
            hard = []
            soft = []
            for c in cannibalizations:
                for comp in c.competing_urls:
                    item = {'query': c.query, 'slug': comp['slug'], 'url': comp['url'], 'type': comp['type'], 
                            'clicks': comp['clicks'], 'main': c.main_url['slug'], 'issue': c.issue_type, 'sev': c.severity.value}
                    if c.issue_type in ["Blog vs PLP", "PLPs duplicadas", "Blogs duplicados"]:
                        hard.append(item)
                    else:
                        soft.append(item)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"🔴 Acciones Hard ({len(hard)})")
                for item in sorted(hard, key=lambda x: x['clicks'], reverse=True)[:12]:
                    st.markdown(f"""
                    <div class="card card-danger">
                        {badge(item['type'], item['type'].lower())}
                        <div style="font-weight:600;margin:8px 0;">301 → /{item['main'][:30]}...</div>
                        <div class="url-slug">/{item['slug'][:40]}...</div>
                        <div style="font-size:12px;margin-top:4px;">Clics: {item['clicks']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.subheader(f"🟡 Acciones Soft ({len(soft)})")
                for item in sorted(soft, key=lambda x: x['clicks'], reverse=True)[:12]:
                    action = "Canonical" if item['issue'] == "PDP vs PLP" else "Diferenciar"
                    st.markdown(f"""
                    <div class="card">
                        {badge(item['type'], item['type'].lower())}
                        <div style="font-weight:600;margin:8px 0;">{action}</div>
                        <div class="url-slug">/{item['slug'][:40]}...</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.divider()
            if st.button("📥 Exportar CSV"):
                export = [{'familia': family, 'query': c.query, 'url': comp['url'], 'tipo': comp['type'],
                          'clics': comp['clicks'], 'issue': c.issue_type, 'severidad': c.severity.value}
                         for c in cannibalizations for comp in c.competing_urls]
                st.download_button("💾 Descargar", pd.DataFrame(export).to_csv(index=False), f"recomendaciones_{family}.csv")


if __name__ == "__main__":
    main()
