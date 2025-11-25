"""
SEO Cannibalization Analyzer v2.0
Herramienta avanzada de análisis de canibalizaciones SEO
Con análisis de arquitectura web nivel SEO Senior
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from urllib.parse import urlparse
import requests
from io import StringIO
import json
import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

# ==================== CONFIGURACIÓN ====================
st.set_page_config(
    page_title="SEO Cannibalization Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS accesible y limpio
st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; max-width: 1400px; }
    
    /* Cards */
    .card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }
    .card-danger { background: #fef2f2; border: 1px solid #fecaca; border-left: 4px solid #dc2626; }
    .card-warning { background: #fffbeb; border: 1px solid #fcd34d; border-left: 4px solid #d97706; }
    .card-success { background: #f0fdf4; border: 1px solid #bbf7d0; border-left: 4px solid #16a34a; }
    .card-info { background: #eff6ff; border: 1px solid #bfdbfe; border-left: 4px solid #2563eb; }
    
    /* Badges */
    .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; margin-right: 4px; }
    .badge-plp { background: #0284c7; color: white; }
    .badge-pdp { background: #7c3aed; color: white; }
    .badge-blog { background: #059669; color: white; }
    .badge-high { background: #dc2626; color: white; }
    .badge-medium { background: #d97706; color: white; }
    .badge-low { background: #16a34a; color: white; }
    .badge-main { background: #2563eb; color: white; }
    
    /* URL items */
    .url-item { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.75rem; margin: 0.5rem 0; }
    .url-item-main { background: #eff6ff; border: 2px solid #2563eb; }
    .url-item-competing { background: #fef2f2; border: 1px solid #fecaca; }
    
    /* Slug display */
    .slug { font-family: 'Courier New', monospace; font-size: 13px; color: #0f172a; word-break: break-all; }
    .slug a { color: #0284c7; text-decoration: none; }
    .slug a:hover { text-decoration: underline; }
    
    /* Query highlight */
    .query-tag { background: #dbeafe; color: #1e40af; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 12px; }
    
    /* Metrics row */
    .metrics-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 0.5rem; }
    .metric-item { font-size: 12px; color: #64748b; }
    .metric-value { font-weight: 600; color: #0f172a; }
    
    /* Analysis sections */
    .analysis-section { background: #f8fafc; border-radius: 8px; padding: 1rem; margin: 1rem 0; }
    .analysis-title { font-weight: 600; color: #0f172a; margin-bottom: 0.5rem; }
    
    /* Warning */
    .warning-box { background: #fffbeb; border: 1px solid #fcd34d; border-radius: 8px; padding: 1rem; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)


# ==================== ENUMS Y DATACLASSES ====================
class UrlType(Enum):
    PLP = "PLP"      # Product Listing Page - categorías con /
    PDP = "PDP"      # Product Detail Page - productos con slug concatenado
    BLOG = "BLOG"    # Posts de blog
    BRAND = "BRAND"  # Páginas de marca
    OTHER = "OTHER"

class Severity(Enum):
    CRITICAL = "CRITICAL"  # 4+ URLs o PLP vs BLOG directo
    HIGH = "HIGH"          # 3 URLs compitiendo
    MEDIUM = "MEDIUM"      # 2 URLs diferentes tipos
    LOW = "LOW"            # 2 URLs mismo tipo

class CannibalizationType(Enum):
    PLP_VS_BLOG = "PLP vs Blog"
    PLP_VS_PDP = "PLP vs PDP"  
    MULTI_PLP = "Múltiples PLPs"
    PDP_GENERIC = "PDP por query genérica"
    BLOG_VS_BLOG = "Blogs duplicados"
    MIXED = "Mixto"

@dataclass
class UrlAnalysis:
    url: str
    slug: str
    url_type: UrlType
    family: str
    depth: int
    top_query: str
    clicks: int
    impressions: int
    position: float
    ctr: float
    is_category_match: bool = False
    adobe_sessions: int = 0

@dataclass 
class Cannibalization:
    query: str
    urls: List[UrlAnalysis]
    url_count: int
    total_clicks: int
    total_impressions: int
    avg_position: float
    position_spread: float
    severity: Severity
    cannib_type: CannibalizationType
    main_url: UrlAnalysis
    competing_urls: List[UrlAnalysis]
    click_concentration: float  # % clicks en URL principal
    intent_mismatch: bool
    recommendation: str

@dataclass
class Recommendation:
    url: str
    slug: str
    url_type: UrlType
    action: str
    priority: str
    description: str
    tactics: List[str]
    impact_score: float


# ==================== FUNCIONES DE CLASIFICACIÓN ====================
def extract_slug(url: str) -> str:
    """Extrae el slug desde el dominio"""
    try:
        parsed = urlparse(url)
        return parsed.path.lstrip('/') or '/'
    except:
        return url

def classify_url_advanced(url: str, categories_df: pd.DataFrame = None) -> Tuple[UrlType, str, int]:
    """
    Clasificación avanzada de URLs para PCComponentes
    
    Reglas:
    - PDP: slug en raíz con múltiples "-" (familia-marca-modelo-specs)
    - PLP: directorios separados por "/" 
    - BLOG: contiene /blog/, /noticias/, /guia/
    - BRAND: página de marca
    """
    try:
        parsed = urlparse(url)
        path = parsed.path.lower().strip('/')
        segments = [s for s in path.split('/') if s]
        
        if not segments:
            return UrlType.PLP, 'home', 0
        
        # Detectar BLOG
        blog_patterns = ['/blog/', '/noticias/', '/guia/', '/articulo/', '/magazine/', '/revista/']
        if any(pattern in url.lower() for pattern in blog_patterns):
            family = segments[1] if len(segments) > 1 else segments[0]
            return UrlType.BLOG, family.replace('-', ' '), len(segments)
        
        # Si hay un solo segmento
        if len(segments) == 1:
            slug = segments[0]
            # Contar guiones para determinar si es PDP o PLP
            hyphen_count = slug.count('-')
            
            # PDP: slug largo con muchos guiones (familia-marca-modelo-specs)
            # Ejemplo: portatil-asus-rog-strix-g15-ryzen-9-rtx-4070
            if hyphen_count >= 3:
                # Extraer familia (primera palabra antes del guión)
                family = slug.split('-')[0]
                return UrlType.PDP, family, 1
            else:
                # PLP de categoría principal
                # Ejemplo: portatiles, monitores, tarjetas-graficas
                family = slug.replace('-', ' ')
                return UrlType.PLP, family, 1
        
        # Múltiples segmentos = PLP con subcategorías
        # Ejemplo: /portatiles/gaming/, /tarjetas-graficas/nvidia/
        family = segments[0].replace('-', ' ')
        
        # Verificar si el último segmento es un PDP dentro de categoría
        last_segment = segments[-1]
        if last_segment.count('-') >= 4:
            return UrlType.PDP, family, len(segments)
        
        return UrlType.PLP, family, len(segments)
        
    except Exception as e:
        return UrlType.OTHER, 'unknown', 0

def analyze_url(row: dict, categories_df: pd.DataFrame = None) -> UrlAnalysis:
    """Analiza una URL completa"""
    url = row.get('url', '')
    slug = extract_slug(url)
    url_type, family, depth = classify_url_advanced(url, categories_df)
    
    clicks = int(row.get('top_query_clicks', 0) or 0)
    impressions = int(row.get('top_query_impressions', 0) or 0)
    position = float(row.get('top_query_position', 0) or 0)
    ctr = (clicks / impressions * 100) if impressions > 0 else 0
    
    # Verificar si coincide con categoría conocida
    is_category = False
    if categories_df is not None and not categories_df.empty:
        if 'Slug' in categories_df.columns:
            is_category = slug in categories_df['Slug'].values
    
    return UrlAnalysis(
        url=url,
        slug=slug,
        url_type=url_type,
        family=family,
        depth=depth,
        top_query=str(row.get('top_query', '')),
        clicks=clicks,
        impressions=impressions,
        position=position,
        ctr=ctr,
        is_category_match=is_category,
        adobe_sessions=int(row.get('adobe_sessions', 0) or 0)
    )


# ==================== ANÁLISIS DE CANIBALIZACIONES (SEO SENIOR) ====================
def detect_cannibalizations_advanced(
    df: pd.DataFrame, 
    categories_df: pd.DataFrame = None,
    min_clicks: int = 5,
    min_impressions: int = 100
) -> List[Cannibalization]:
    """
    Detección avanzada de canibalizaciones - Nivel SEO Senior
    
    Criterios de análisis:
    1. Misma top_query en múltiples URLs
    2. Análisis de intent mismatch (PLP vs Blog)
    3. Concentración de clics (si está muy distribuido = problema)
    4. Spread de posiciones (varianza alta = confusión de Google)
    5. Profundidad vs rendimiento
    """
    cannibalizations = []
    
    # Agrupar por top_query
    query_groups = df.groupby('top_query')
    
    for query, group in query_groups:
        # Filtrar queries vacías o con poco volumen
        if pd.isna(query) or not str(query).strip():
            continue
        
        total_clicks = group['top_query_clicks'].sum()
        total_impressions = group['top_query_impressions'].sum()
        
        if total_clicks < min_clicks or total_impressions < min_impressions:
            continue
        
        if len(group) < 2:
            continue
        
        # Analizar cada URL
        url_analyses = []
        for _, row in group.iterrows():
            analysis = analyze_url(row.to_dict(), categories_df)
            url_analyses.append(analysis)
        
        # Ordenar por clics (descendente)
        url_analyses.sort(key=lambda x: x.clicks, reverse=True)
        
        main_url = url_analyses[0]
        competing_urls = url_analyses[1:]
        
        # Calcular métricas de canibalización
        positions = [u.position for u in url_analyses if u.position > 0]
        avg_position = np.mean(positions) if positions else 0
        position_spread = max(positions) - min(positions) if len(positions) > 1 else 0
        
        click_concentration = (main_url.clicks / total_clicks * 100) if total_clicks > 0 else 0
        
        # Determinar tipo de canibalización
        url_types = set(u.url_type for u in url_analyses)
        
        if UrlType.PLP in url_types and UrlType.BLOG in url_types:
            cannib_type = CannibalizationType.PLP_VS_BLOG
        elif UrlType.PLP in url_types and UrlType.PDP in url_types:
            cannib_type = CannibalizationType.PLP_VS_PDP
        elif len([u for u in url_analyses if u.url_type == UrlType.PLP]) > 1:
            cannib_type = CannibalizationType.MULTI_PLP
        elif main_url.url_type == UrlType.PDP and len(str(query).split()) <= 2:
            cannib_type = CannibalizationType.PDP_GENERIC
        elif all(u.url_type == UrlType.BLOG for u in url_analyses):
            cannib_type = CannibalizationType.BLOG_VS_BLOG
        else:
            cannib_type = CannibalizationType.MIXED
        
        # Detectar intent mismatch
        intent_mismatch = (
            (UrlType.PLP in url_types and UrlType.BLOG in url_types) or
            (main_url.url_type == UrlType.PDP and cannib_type == CannibalizationType.PDP_GENERIC)
        )
        
        # Determinar severidad
        url_count = len(url_analyses)
        if url_count >= 4 or (cannib_type == CannibalizationType.PLP_VS_BLOG and click_concentration < 60):
            severity = Severity.CRITICAL
        elif url_count >= 3 or intent_mismatch:
            severity = Severity.HIGH
        elif url_count == 2 and len(url_types) > 1:
            severity = Severity.MEDIUM
        else:
            severity = Severity.LOW
        
        # Generar recomendación automática
        recommendation = generate_auto_recommendation(
            cannib_type, main_url, competing_urls, click_concentration, position_spread
        )
        
        cannibalizations.append(Cannibalization(
            query=str(query),
            urls=url_analyses,
            url_count=url_count,
            total_clicks=int(total_clicks),
            total_impressions=int(total_impressions),
            avg_position=avg_position,
            position_spread=position_spread,
            severity=severity,
            cannib_type=cannib_type,
            main_url=main_url,
            competing_urls=competing_urls,
            click_concentration=click_concentration,
            intent_mismatch=intent_mismatch,
            recommendation=recommendation
        ))
    
    # Ordenar por severidad y clics
    severity_order = {Severity.CRITICAL: 4, Severity.HIGH: 3, Severity.MEDIUM: 2, Severity.LOW: 1}
    cannibalizations.sort(key=lambda x: (severity_order[x.severity], x.total_clicks), reverse=True)
    
    return cannibalizations

def generate_auto_recommendation(
    cannib_type: CannibalizationType,
    main_url: UrlAnalysis,
    competing_urls: List[UrlAnalysis],
    click_concentration: float,
    position_spread: float
) -> str:
    """Genera recomendación automática basada en el tipo de canibalización"""
    
    if cannib_type == CannibalizationType.PLP_VS_BLOG:
        if main_url.url_type == UrlType.PLP:
            return "✅ PLP debe ser la URL principal. Añadir canonical en Blog hacia PLP o redirigir 301 si el blog no aporta valor diferencial."
        else:
            return "⚠️ Blog rankeando mejor que PLP comercial. Revisar contenido de PLP y considerar fusionar contenidos."
    
    elif cannib_type == CannibalizationType.PLP_VS_PDP:
        return "📦 PDP compitiendo por query genérica. Implementar canonical hacia PLP padre y optimizar PDP para long-tail específico."
    
    elif cannib_type == CannibalizationType.MULTI_PLP:
        return "🔀 Múltiples PLPs para misma query. Consolidar en una única PLP y redirigir 301 las secundarias."
    
    elif cannib_type == CannibalizationType.PDP_GENERIC:
        return "🎯 PDP posicionando por query genérica. Reforzar PLP padre con enlazado interno y diferenciar contenido de PDP."
    
    elif cannib_type == CannibalizationType.BLOG_VS_BLOG:
        return "📝 Posts duplicados. Fusionar contenidos en el mejor artículo y redirigir 301 el resto."
    
    else:
        if click_concentration < 50:
            return "⚡ Clics muy distribuidos. Consolidar autoridad en URL principal con enlazado interno agresivo."
        elif position_spread > 10:
            return "📊 Alta varianza de posiciones. Google confundido sobre URL canónica. Implementar señales claras."
        else:
            return "🔍 Analizar caso individualmente. Posible diferenciación de intent necesaria."


def generate_recommendations_advanced(cannibalization: Cannibalization) -> List[Recommendation]:
    """Genera recomendaciones detalladas para cada URL"""
    recommendations = []
    
    # URL Principal
    main = cannibalization.main_url
    recommendations.append(Recommendation(
        url=main.url,
        slug=main.slug,
        url_type=main.url_type,
        action="MANTENER Y POTENCIAR",
        priority="ALTA",
        description=f"URL principal con {cannibalization.click_concentration:.0f}% de los clics. Reforzar como destino canónico.",
        tactics=[
            "Reforzar enlazado interno desde páginas de alta autoridad",
            "Optimizar meta title incluyendo la query principal",
            "Añadir contenido complementario (FAQs, comparativas)",
            "Verificar que H1 incluye la keyword principal",
            "Implementar Schema markup apropiado"
        ],
        impact_score=0.9
    ))
    
    # URLs competidoras
    for comp in cannibalization.competing_urls:
        action, priority, description, tactics = get_competing_url_recommendation(
            comp, main, cannibalization
        )
        
        recommendations.append(Recommendation(
            url=comp.url,
            slug=comp.slug,
            url_type=comp.url_type,
            action=action,
            priority=priority,
            description=description,
            tactics=tactics,
            impact_score=0.7 if priority == "ALTA" else 0.5 if priority == "MEDIA" else 0.3
        ))
    
    return recommendations

def get_competing_url_recommendation(
    comp: UrlAnalysis,
    main: UrlAnalysis,
    cannib: Cannibalization
) -> Tuple[str, str, str, List[str]]:
    """Determina la recomendación para una URL competidora"""
    
    # Blog compitiendo con PLP
    if comp.url_type == UrlType.BLOG and main.url_type == UrlType.PLP:
        return (
            "REDIRIGIR 301 / CANONICAL",
            "ALTA",
            "Blog compitiendo con PLP transaccional. El intent comercial debe resolverse en PLP.",
            [
                "Opción A: Redirigir 301 hacia la PLP (si el blog no tiene backlinks valiosos)",
                "Opción B: Mantener blog con canonical hacia PLP",
                "Opción C: Mover a /blog/guia-{topic} con enfoque informacional diferente",
                "Actualizar enlaces internos que apuntan al blog",
                "Eliminar del sitemap XML tras redirección"
            ]
        )
    
    # PDP compitiendo por query genérica
    elif comp.url_type == UrlType.PDP and main.url_type == UrlType.PLP:
        return (
            "CANONICAL + DIFERENCIAR",
            "MEDIA",
            "PDP rankeando por query de categoría. Diferenciar hacia long-tail.",
            [
                "Implementar rel='canonical' hacia la PLP padre",
                "Reoptimizar title/H1 hacia keywords específicas del producto",
                "Añadir contenido único: specs, reviews, comparativas",
                "Verificar que breadcrumbs apuntan correctamente a PLP",
                "Considerar noindex si hay muchos PDPs similares"
            ]
        )
    
    # Múltiples PLPs
    elif comp.url_type == UrlType.PLP and main.url_type == UrlType.PLP:
        if comp.clicks < main.clicks * 0.2:  # Menos del 20% de clics
            return (
                "REDIRIGIR 301",
                "ALTA",
                "PLP secundaria con bajo rendimiento. Consolidar autoridad en PLP principal.",
                [
                    "Implementar redirección 301 hacia PLP principal",
                    "Migrar cualquier contenido único antes de redirigir",
                    "Actualizar enlaces internos",
                    "Notificar cambio en Search Console",
                    "Monitorizar pérdida de posiciones durante 4 semanas"
                ]
            )
        else:
            return (
                "DIFERENCIAR INTENT",
                "MEDIA",
                "PLPs con rendimiento similar. Evaluar si responden a intents diferentes.",
                [
                    "Analizar queries secundarias de cada PLP",
                    "Si mismo intent: fusionar en una sola PLP",
                    "Si diferente intent: diferenciar titles y contenido",
                    "Establecer enlazado cruzado si son complementarias",
                    "Revisar arquitectura de categorías"
                ]
            )
    
    # Blog vs Blog
    elif comp.url_type == UrlType.BLOG and main.url_type == UrlType.BLOG:
        return (
            "FUSIONAR CONTENIDO",
            "MEDIA",
            "Posts duplicados o muy similares. Consolidar en un único artículo.",
            [
                "Fusionar contenido en el post con mejor rendimiento",
                "Redirigir 301 el post secundario",
                "Actualizar fecha de publicación tras fusión",
                "Añadir secciones nuevas para mejorar completitud",
                "Promocionar el post consolidado"
            ]
        )
    
    # Caso por defecto
    else:
        click_ratio = comp.clicks / max(cannib.total_clicks, 1)
        if click_ratio < 0.1:
            return (
                "NOINDEX / ELIMINAR",
                "BAJA",
                "URL con rendimiento marginal. Evaluar eliminación.",
                [
                    "Si tiene contenido único: aplicar noindex",
                    "Si es thin content: eliminar con 410",
                    "Si tiene backlinks: redirigir 301",
                    "Excluir del sitemap XML",
                    "Bloquear en robots.txt temporalmente para test"
                ]
            )
        else:
            return (
                "ANALIZAR MANUALMENTE",
                "MEDIA",
                "Caso requiere análisis individual del intent y contenido.",
                [
                    "Revisar queries secundarias de cada URL",
                    "Analizar comportamiento de usuario (Analytics)",
                    "Evaluar si hay diferenciación real de intent",
                    "Considerar prueba A/B de consolidación",
                    "Consultar con equipo de contenido"
                ]
            )


# ==================== VISUALIZACIONES ====================
def create_cannibalization_overview(cannibalizations: List[Cannibalization]) -> go.Figure:
    """Crea gráfico resumen de canibalizaciones"""
    
    # Por severidad
    severity_counts = {
        'Crítica': len([c for c in cannibalizations if c.severity == Severity.CRITICAL]),
        'Alta': len([c for c in cannibalizations if c.severity == Severity.HIGH]),
        'Media': len([c for c in cannibalizations if c.severity == Severity.MEDIUM]),
        'Baja': len([c for c in cannibalizations if c.severity == Severity.LOW])
    }
    
    # Por tipo
    type_counts = defaultdict(int)
    for c in cannibalizations:
        type_counts[c.cannib_type.value] += 1
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Por Severidad', 'Por Tipo'),
        specs=[[{"type": "bar"}, {"type": "pie"}]]
    )
    
    # Bar chart severidad
    fig.add_trace(
        go.Bar(
            x=list(severity_counts.keys()),
            y=list(severity_counts.values()),
            marker_color=['#7f1d1d', '#dc2626', '#d97706', '#16a34a'],
            text=list(severity_counts.values()),
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Pie chart tipo
    fig.add_trace(
        go.Pie(
            labels=list(type_counts.keys()),
            values=list(type_counts.values()),
            marker_colors=['#0284c7', '#7c3aed', '#059669', '#d97706', '#dc2626', '#64748b']
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=350,
        showlegend=True,
        plot_bgcolor='#f8fafc',
        paper_bgcolor='#ffffff',
        font=dict(color='#0f172a')
    )
    
    return fig

def create_linking_graph(df: pd.DataFrame, family: str) -> go.Figure:
    """Crea grafo de enlaces internos"""
    family_df = df[df['url'].str.lower().str.contains(family.lower(), na=False)].head(20)
    
    if len(family_df) == 0:
        return None
    
    G = nx.DiGraph()
    
    nodes_by_type = {'PLP': [], 'PDP': [], 'BLOG': []}
    
    for _, row in family_df.iterrows():
        url = row['url']
        slug = extract_slug(url)[:30]
        url_type, _, depth = classify_url_advanced(url)
        
        G.add_node(slug, type=url_type.value, full_url=url)
        nodes_by_type.get(url_type.value, []).append(slug)
    
    # Crear enlaces sugeridos
    plps = nodes_by_type.get('PLP', [])
    pdps = nodes_by_type.get('PDP', [])
    blogs = nodes_by_type.get('BLOG', [])
    
    if plps:
        main_plp = plps[0]
        for pdp in pdps[:8]:
            G.add_edge(main_plp, pdp)
        for blog in blogs[:3]:
            G.add_edge(main_plp, blog)
            G.add_edge(blog, main_plp)  # Bidireccional
    
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    fig = go.Figure()
    
    # Edges
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(width=1.5, color='#94a3b8'),
            hoverinfo='none',
            showlegend=False
        ))
    
    # Nodes
    colors = {'PLP': '#0284c7', 'PDP': '#7c3aed', 'BLOG': '#059669'}
    
    for node_type, color in colors.items():
        nodes = [n for n in G.nodes() if G.nodes[n].get('type') == node_type]
        if nodes:
            fig.add_trace(go.Scatter(
                x=[pos[n][0] for n in nodes],
                y=[pos[n][1] for n in nodes],
                mode='markers+text',
                marker=dict(size=35, color=color),
                text=[n[:15] for n in nodes],
                textposition="bottom center",
                textfont=dict(size=9, color='#0f172a'),
                name=node_type,
                hovertext=[G.nodes[n].get('full_url', n) for n in nodes]
            ))
    
    fig.update_layout(
        title="Propuesta de Enlazado Interno",
        showlegend=True,
        plot_bgcolor='#f8fafc',
        paper_bgcolor='#ffffff',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500
    )
    
    return fig


# ==================== HELPERS HTML ====================
def badge(text: str, variant: str) -> str:
    return f'<span class="badge badge-{variant}">{text}</span>'

def format_url_item(analysis: UrlAnalysis, is_main: bool = False, show_query: bool = True) -> str:
    """Formatea un item de URL para mostrar"""
    
    type_badge = badge(analysis.url_type.value, analysis.url_type.value.lower())
    main_badge = badge("PRINCIPAL", "main") if is_main else ""
    
    query_html = ""
    if show_query and analysis.top_query:
        query_html = f'<span class="query-tag">{analysis.top_query[:50]}</span>'
    
    item_class = "url-item-main" if is_main else "url-item"
    
    return f"""
    <div class="{item_class}">
        <div>{type_badge} {main_badge}</div>
        <div class="slug"><a href="{analysis.url}" target="_blank">/{analysis.slug}</a></div>
        {f'<div style="margin-top:4px">{query_html}</div>' if query_html else ''}
        <div class="metrics-row">
            <span class="metric-item">Clics: <span class="metric-value">{analysis.clicks:,}</span></span>
            <span class="metric-item">Impr: <span class="metric-value">{analysis.impressions:,}</span></span>
            <span class="metric-item">Pos: <span class="metric-value">{analysis.position:.1f}</span></span>
            <span class="metric-item">CTR: <span class="metric-value">{analysis.ctr:.2f}%</span></span>
            {f'<span class="metric-item">Sessions: <span class="metric-value">{analysis.adobe_sessions:,}</span></span>' if analysis.adobe_sessions > 0 else ''}
        </div>
    </div>
    """


# ==================== INTERFAZ PRINCIPAL ====================
def main():
    st.title("🎯 SEO Cannibalization Analyzer")
    st.caption("Análisis avanzado de canibalizaciones · Arquitectura web · Nivel SEO Senior")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Archivos de datos")
        
        # CSV principal Search Console
        st.subheader("1. Search Console (obligatorio)")
        gsc_file = st.file_uploader(
            "CSV de Search Console",
            type=['csv'],
            key="gsc",
            help="Exporta desde Search Console con columnas: url, top_query, clicks, impressions, position"
        )
        
        # CSV categorías
        st.subheader("2. Categorías (opcional)")
        cat_file = st.file_uploader(
            "CSV de Categorías",
            type=['csv'],
            key="categories",
            help="Archivo con slugs de categorías principales"
        )
        
        # CSV Adobe Analytics
        st.subheader("3. Adobe Analytics (opcional)")
        adobe_file = st.file_uploader(
            "CSV de Adobe Analytics",
            type=['csv'],
            key="adobe",
            help="Datos de tráfico orgánico con columnas: url, sessions"
        )
        
        st.divider()
        
        st.header("⚙️ Configuración")
        
        min_clicks = st.slider("Mín. clics", 0, 100, 5)
        min_impressions = st.slider("Mín. impresiones", 0, 1000, 100)
        
        st.divider()
        
        st.header("🔑 APIs (opcional)")
        anthropic_key = st.text_input("Anthropic API Key", type="password")
        openai_key = st.text_input("OpenAI API Key", type="password")
        semrush_key = st.text_input("Semrush API Key", type="password")
    
    # Procesar archivos
    df = None
    categories_df = None
    adobe_df = None
    
    if gsc_file:
        df = pd.read_csv(gsc_file)
        st.session_state['df'] = df
        
        if cat_file:
            categories_df = pd.read_csv(cat_file)
            st.session_state['categories_df'] = categories_df
        
        if adobe_file:
            adobe_df = pd.read_csv(adobe_file)
            # Intentar merge con datos principales
            if 'url' in adobe_df.columns and 'sessions' in adobe_df.columns:
                df = df.merge(
                    adobe_df[['url', 'sessions']].rename(columns={'sessions': 'adobe_sessions'}),
                    on='url',
                    how='left'
                )
                df['adobe_sessions'] = df['adobe_sessions'].fillna(0)
                st.session_state['df'] = df
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 Datos",
        "🔍 Análisis",
        "💡 Recomendaciones",
        "🔗 Arquitectura",
        "📈 Competencia"
    ])
    
    # ==================== TAB 1: DATOS ====================
    with tab1:
        if df is None:
            st.info("👈 Sube tu archivo CSV de Search Console en la barra lateral")
            
            with st.expander("📋 Formato esperado del CSV"):
                st.markdown("""
                **Columnas requeridas:**
                - `url` - URL completa
                - `top_query` - Query principal
                - `top_query_clicks` - Clics de la query
                - `top_query_impressions` - Impresiones
                - `top_query_position` - Posición media
                
                **Columnas opcionales:**
                - `url_total_clicks`, `url_total_impressions`, `url_avg_position`
                - `num_queries_conocidas`, `pct_clicks_from_top_query_all`
                """)
        else:
            st.success(f"✅ Datos cargados: **{len(df):,} filas**")
            
            # Clasificar URLs
            if 'url_type' not in df.columns:
                df['url_type'] = df['url'].apply(lambda x: classify_url_advanced(x, categories_df)[0].value)
                df['family'] = df['url'].apply(lambda x: classify_url_advanced(x, categories_df)[1])
                df['slug'] = df['url'].apply(extract_slug)
                st.session_state['df'] = df
            
            # Métricas
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("URLs", f"{len(df):,}")
            with col2:
                st.metric("Clics", f"{df['top_query_clicks'].sum():,}")
            with col3:
                st.metric("Impresiones", f"{df['top_query_impressions'].sum():,}")
            with col4:
                st.metric("PLPs", len(df[df['url_type'] == 'PLP']))
            with col5:
                st.metric("PDPs", len(df[df['url_type'] == 'PDP']))
            
            # Distribución
            col1, col2 = st.columns(2)
            
            with col1:
                type_counts = df['url_type'].value_counts()
                fig = px.pie(values=type_counts.values, names=type_counts.index,
                           title="Distribución por tipo", 
                           color_discrete_sequence=['#0284c7', '#7c3aed', '#059669', '#64748b'])
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                family_counts = df['family'].value_counts().head(10)
                fig = px.bar(x=family_counts.values, y=family_counts.index, orientation='h',
                           title="Top 10 Familias", color_discrete_sequence=['#0284c7'])
                fig.update_layout(height=300, yaxis=dict(title=""))
                st.plotly_chart(fig, use_container_width=True)
            
            # Preview
            st.subheader("Vista previa")
            st.dataframe(
                df[['slug', 'url_type', 'family', 'top_query', 'top_query_clicks', 'top_query_impressions', 'top_query_position']].head(20),
                use_container_width=True
            )
    
    # ==================== TAB 2: ANÁLISIS ====================
    with tab2:
        if df is None:
            st.warning("⚠️ Primero carga los datos en la pestaña **Datos**")
        else:
            st.header("🔍 Análisis de Canibalizaciones")
            
            # Selector de familia
            col1, col2 = st.columns([3, 1])
            with col1:
                families = df['family'].value_counts()
                family_options = ["Todas las familias"] + [f"{f} ({c} URLs)" for f, c in families.head(50).items()]
                selected = st.selectbox("Familia de productos", family_options)
            
            with col2:
                custom_family = st.text_input("O buscar:")
            
            # Determinar familia
            if custom_family:
                selected_family = custom_family.lower()
            elif selected != "Todas las familias":
                selected_family = selected.split(' (')[0]
            else:
                selected_family = None
            
            # Filtrar datos
            if selected_family:
                analysis_df = df[
                    (df['url'].str.lower().str.contains(selected_family, na=False)) |
                    (df['top_query'].str.lower().str.contains(selected_family, na=False)) |
                    (df['family'].str.lower().str.contains(selected_family, na=False))
                ]
            else:
                analysis_df = df
            
            # Detectar canibalizaciones
            cannibalizations = detect_cannibalizations_advanced(
                analysis_df, 
                categories_df,
                min_clicks=min_clicks,
                min_impressions=min_impressions
            )
            
            st.session_state['cannibalizations'] = cannibalizations
            st.session_state['selected_family'] = selected_family
            
            if cannibalizations:
                # Resumen
                st.subheader(f"📊 Resumen: {len(cannibalizations)} canibalizaciones detectadas")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    critical = len([c for c in cannibalizations if c.severity == Severity.CRITICAL])
                    st.metric("🔴 Críticas", critical)
                with col2:
                    high = len([c for c in cannibalizations if c.severity == Severity.HIGH])
                    st.metric("🟠 Altas", high)
                with col3:
                    medium = len([c for c in cannibalizations if c.severity == Severity.MEDIUM])
                    st.metric("🟡 Medias", medium)
                with col4:
                    low = len([c for c in cannibalizations if c.severity == Severity.LOW])
                    st.metric("🟢 Bajas", low)
                
                # Gráficos
                fig = create_cannibalization_overview(cannibalizations)
                st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                
                # Filtro de severidad para el desglose
                st.subheader("📋 Desglose de Canibalizaciones")
                
                severity_options = ["Todas", "Crítica", "Alta", "Media", "Baja"]
                selected_severity = st.radio(
                    "Filtrar por severidad:",
                    severity_options,
                    horizontal=True
                )
                
                # Mapeo de severidad
                severity_map = {
                    "Crítica": Severity.CRITICAL,
                    "Alta": Severity.HIGH,
                    "Media": Severity.MEDIUM,
                    "Baja": Severity.LOW
                }
                
                # Filtrar canibalizaciones
                if selected_severity != "Todas":
                    filtered_cannib = [c for c in cannibalizations if c.severity == severity_map[selected_severity]]
                else:
                    filtered_cannib = cannibalizations
                
                st.caption(f"Mostrando {len(filtered_cannib)} de {len(cannibalizations)} canibalizaciones")
                
                # Lista de canibalizaciones
                for cannib in filtered_cannib[:30]:
                    severity_emoji = {
                        Severity.CRITICAL: "🔴",
                        Severity.HIGH: "🟠",
                        Severity.MEDIUM: "🟡",
                        Severity.LOW: "🟢"
                    }[cannib.severity]
                    
                    severity_badge_class = {
                        Severity.CRITICAL: "high",
                        Severity.HIGH: "high",
                        Severity.MEDIUM: "medium",
                        Severity.LOW: "low"
                    }[cannib.severity]
                    
                    with st.expander(f"{severity_emoji} `{cannib.query}` — {cannib.url_count} URLs · {cannib.total_clicks:,} clics"):
                        # Info de canibalización
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Tipo", cannib.cannib_type.value)
                        with col2:
                            st.metric("Concentración", f"{cannib.click_concentration:.0f}%")
                        with col3:
                            st.metric("Spread posición", f"{cannib.position_spread:.1f}")
                        with col4:
                            st.metric("Pos. media", f"{cannib.avg_position:.1f}")
                        
                        # Recomendación automática
                        st.markdown(f"""
                        <div class="card card-info">
                            <strong>💡 Recomendación:</strong> {cannib.recommendation}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # URLs
                        st.markdown("**URLs involucradas:**")
                        
                        # URL principal
                        st.markdown(format_url_item(cannib.main_url, is_main=True, show_query=True), unsafe_allow_html=True)
                        
                        # URLs competidoras
                        for comp in cannib.competing_urls:
                            st.markdown(format_url_item(comp, is_main=False, show_query=True), unsafe_allow_html=True)
                
            else:
                st.success("✅ No se detectaron canibalizaciones significativas")
            
            # Análisis con IA
            if cannibalizations:
                st.divider()
                st.subheader("🤖 Análisis con IA")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔮 Analizar con Claude", disabled=not anthropic_key, use_container_width=True):
                        with st.spinner("Analizando..."):
                            # Preparar datos
                            data = [{
                                "query": c.query,
                                "urls": c.url_count,
                                "clicks": c.total_clicks,
                                "type": c.cannib_type.value,
                                "severity": c.severity.value,
                                "recommendation": c.recommendation
                            } for c in cannibalizations[:10]]
                            
                            result = analyze_with_ai(data, anthropic_key, "anthropic")
                            st.markdown(result)
                
                with col2:
                    if st.button("🧠 Analizar con GPT-4", disabled=not openai_key, use_container_width=True):
                        with st.spinner("Analizando..."):
                            data = [{
                                "query": c.query,
                                "urls": c.url_count,
                                "clicks": c.total_clicks,
                                "type": c.cannib_type.value,
                                "severity": c.severity.value
                            } for c in cannibalizations[:10]]
                            
                            result = analyze_with_ai(data, openai_key, "openai")
                            st.markdown(result)
    
    # ==================== TAB 3: RECOMENDACIONES ====================
    with tab3:
        if 'cannibalizations' not in st.session_state or not st.session_state['cannibalizations']:
            st.warning("⚠️ Primero ejecuta el análisis en la pestaña **Análisis**")
        else:
            cannibalizations = st.session_state['cannibalizations']
            
            st.header("💡 Recomendaciones SEO")
            
            st.markdown("""
            <div class="warning-box">
                ⚠️ <strong>Disclaimer:</strong> Todas las recomendaciones deben validarse con el Departamento SEO antes de implementar.
                Los cambios estructurales (301, 410, noindex) requieren análisis de impacto previo.
            </div>
            """, unsafe_allow_html=True)
            
            # Generar todas las recomendaciones
            all_recs = []
            for cannib in cannibalizations[:15]:
                recs = generate_recommendations_advanced(cannib)
                all_recs.extend([(cannib.query, r) for r in recs])
            
            # Separar por tipo
            hard_recs = [(q, r) for q, r in all_recs if r.action in ["REDIRIGIR 301 / CANONICAL", "NOINDEX / ELIMINAR", "REDIRIGIR 301", "FUSIONAR CONTENIDO"]]
            soft_recs = [(q, r) for q, r in all_recs if r.action not in ["REDIRIGIR 301 / CANONICAL", "NOINDEX / ELIMINAR", "REDIRIGIR 301", "FUSIONAR CONTENIDO"]]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔨 Acciones Hard")
                st.caption("Cambios estructurales: redirecciones, eliminaciones")
                
                for query, rec in hard_recs[:15]:
                    priority_color = "high" if rec.priority == "ALTA" else "medium" if rec.priority == "MEDIA" else "low"
                    st.markdown(f"""
                    <div class="card card-danger">
                        <div>{badge(rec.action, 'high')} {badge(rec.url_type.value, rec.url_type.value.lower())} {badge(f"P:{rec.priority}", priority_color)}</div>
                        <div class="slug" style="margin: 8px 0;"><a href="{rec.url}" target="_blank">/{rec.slug}</a></div>
                        <p style="margin: 8px 0; font-size: 13px;">{rec.description}</p>
                        <small style="color: #64748b;">Query: {query}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("Ver tácticas", expanded=False):
                        for tactic in rec.tactics:
                            st.markdown(f"• {tactic}")
            
            with col2:
                st.subheader("🎯 Acciones Soft")
                st.caption("Optimización: canonical, diferenciación, enlazado")
                
                for query, rec in soft_recs[:15]:
                    priority_color = "high" if rec.priority == "ALTA" else "medium" if rec.priority == "MEDIA" else "low"
                    card_class = "card-success" if rec.action == "MANTENER Y POTENCIAR" else "card"
                    st.markdown(f"""
                    <div class="card {card_class}">
                        <div>{badge(rec.action, 'low' if rec.action == "MANTENER Y POTENCIAR" else 'medium')} {badge(rec.url_type.value, rec.url_type.value.lower())} {badge(f"P:{rec.priority}", priority_color)}</div>
                        <div class="slug" style="margin: 8px 0;"><a href="{rec.url}" target="_blank">/{rec.slug}</a></div>
                        <p style="margin: 8px 0; font-size: 13px;">{rec.description}</p>
                        <small style="color: #64748b;">Query: {query}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("Ver tácticas", expanded=False):
                        for tactic in rec.tactics:
                            st.markdown(f"• {tactic}")
            
            # Exportar
            st.divider()
            if st.button("📥 Exportar recomendaciones a CSV", use_container_width=True):
                export_data = [{
                    'query': q,
                    'url': r.url,
                    'slug': r.slug,
                    'tipo': r.url_type.value,
                    'accion': r.action,
                    'prioridad': r.priority,
                    'descripcion': r.description,
                    'tacticas': ' | '.join(r.tactics)
                } for q, r in all_recs]
                
                csv = pd.DataFrame(export_data).to_csv(index=False)
                st.download_button("💾 Descargar CSV", csv, "recomendaciones_seo.csv", "text/csv")
    
    # ==================== TAB 4: ARQUITECTURA ====================
    with tab4:
        if df is None:
            st.warning("⚠️ Primero carga los datos")
        else:
            st.header("🔗 Arquitectura de Enlaces Internos")
            
            family = st.session_state.get('selected_family', '')
            if not family:
                family = st.text_input("Introduce familia a analizar:", value="portatiles")
            
            if family:
                fig = create_linking_graph(df, family)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="card card-success">
                        <h4>✅ Buenas Prácticas</h4>
                        <ul style="margin: 0; padding-left: 20px;">
                            <li>PLP principal enlaza a todos los PDPs hijos</li>
                            <li>Blog enlaza a PLP (no a PDPs)</li>
                            <li>Breadcrumbs consistentes</li>
                            <li>Máximo 3 clics desde home</li>
                            <li>Anchor text descriptivo con keywords</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="card card-danger">
                        <h4>❌ Evitar</h4>
                        <ul style="margin: 0; padding-left: 20px;">
                            <li>PDPs huérfanos sin enlace desde PLP</li>
                            <li>Enlaces cruzados entre PDPs</li>
                            <li>Más de 100 enlaces por página</li>
                            <li>Anchor text genérico ("ver más")</li>
                            <li>Profundidad > 4 niveles</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
    
    # ==================== TAB 5: COMPETENCIA ====================
    with tab5:
        st.header("📈 Análisis de Competencia")
        
        if not semrush_key:
            st.info("🔑 Añade tu API Key de Semrush en la barra lateral")
        else:
            keyword = st.text_input("Keyword a analizar")
            
            if keyword and st.button("🔍 Analizar Top 5"):
                with st.spinner("Consultando Semrush..."):
                    results = get_semrush_data(keyword, semrush_key)
                    if results is not None and not results.empty:
                        st.dataframe(results, use_container_width=True)
                    else:
                        st.warning("No se encontraron resultados")
        
        st.divider()
        st.subheader("🔬 Análisis manual de URL")
        
        comp_url = st.text_input("URL de competidor:")
        if comp_url:
            url_type, family, depth = classify_url_advanced(comp_url)
            slug = extract_slug(comp_url)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tipo", url_type.value)
            with col2:
                st.metric("Familia", family)
            with col3:
                st.metric("Profundidad", depth)
            
            st.code(f"Slug: /{slug}")


def analyze_with_ai(data: list, api_key: str, provider: str) -> str:
    """Analiza datos con IA"""
    prompt = f"""Como SEO Senior especializado en arquitectura web y canibalizaciones, analiza estos casos:

{json.dumps(data, indent=2, ensure_ascii=False)}

Proporciona:
1. Resumen ejecutivo (3-4 líneas)
2. Top 3 acciones prioritarias
3. Impacto estimado
4. Riesgos a considerar

Responde en español, de forma concisa y accionable."""

    try:
        if provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        else:
            import openai
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                max_tokens=2000,
                messages=[
                    {"role": "system", "content": "Eres un SEO Senior experto en arquitectura web."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

def get_semrush_data(keyword: str, api_key: str) -> pd.DataFrame:
    """Obtiene datos de Semrush"""
    try:
        params = {
            "type": "phrase_organic",
            "key": api_key,
            "phrase": keyword,
            "database": "es",
            "display_limit": 5
        }
        response = requests.get("https://api.semrush.com/", params=params)
        if response.status_code == 200:
            return pd.read_csv(StringIO(response.text), sep=';')
        return pd.DataFrame()
    except:
        return pd.DataFrame()


if __name__ == "__main__":
    main()
