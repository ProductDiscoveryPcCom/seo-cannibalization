"""
SEO Cannibalization Analyzer
Herramienta de análisis de canibalizaciones SEO para PCComponentes
Deploy: Streamlit Cloud
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
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# ==================== CONFIGURACIÓN ====================
st.set_page_config(
    page_title="SEO Cannibalization Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS mínimo y accesible
st.markdown("""
<style>
    /* Mejoras sutiles manteniendo accesibilidad */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* Cards para recomendaciones */
    .recommendation-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .recommendation-card-hard {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 4px solid #dc2626;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .recommendation-card-soft {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 4px solid #16a34a;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Badges accesibles */
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 4px;
    }
    
    .badge-plp {
        background: #0284c7;
        color: white;
    }
    
    .badge-pdp {
        background: #7c3aed;
        color: white;
    }
    
    .badge-blog {
        background: #059669;
        color: white;
    }
    
    .badge-high {
        background: #dc2626;
        color: white;
    }
    
    .badge-medium {
        background: #d97706;
        color: white;
    }
    
    .badge-low {
        background: #16a34a;
        color: white;
    }
    
    /* URL list items */
    .url-item {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    
    .url-item-main {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-left: 4px solid #2563eb;
        border-radius: 6px;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    
    /* Warning box */
    .warning-box {
        background: #fffbeb;
        border: 1px solid #fcd34d;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Severity boxes */
    .severity-high {
        background: #fef2f2;
        border-left: 4px solid #dc2626;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    
    .severity-medium {
        background: #fffbeb;
        border-left: 4px solid #d97706;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    
    .severity-low {
        background: #f0fdf4;
        border-left: 4px solid #16a34a;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    
    /* Links accesibles */
    a {
        color: #0284c7;
        text-decoration: underline;
    }
    
    a:hover {
        color: #0369a1;
    }
    
    /* Info boxes */
    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
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
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

@dataclass
class Cannibalization:
    query: str
    urls: List[Dict]
    url_count: int
    total_clicks: int
    total_impressions: int
    position_variance: float
    severity: Severity
    main_url: str
    competing_urls: List[str]

@dataclass
class Recommendation:
    url: str
    url_type: UrlType
    action: str
    priority: str
    description: str
    tactics: List[str]


# ==================== FUNCIONES DE UTILIDAD ====================
def classify_url(url: str) -> UrlType:
    """Clasifica una URL en PLP, PDP o BLOG"""
    try:
        pathname = urlparse(url).path.lower()
        
        # Detectar Blog
        blog_patterns = ['/blog/', '/noticias/', '/guia/', '/articulo/', '/magazine/']
        if any(pattern in pathname for pattern in blog_patterns):
            return UrlType.BLOG
        
        segments = [s for s in pathname.split('/') if s]
        
        if not segments:
            return UrlType.PLP
        
        last_segment = segments[-1]
        
        # PDP: slugs largos con múltiples atributos
        if '-' in last_segment and len(last_segment.split('-')) > 4:
            return UrlType.PDP
        
        # PLP: categorías con pocos niveles
        if 1 <= len(segments) <= 3:
            return UrlType.PLP
        
        return UrlType.PLP
        
    except Exception:
        return UrlType.OTHER

def extract_family_from_url(url: str) -> str:
    """Extrae la familia de producto de una URL"""
    try:
        pathname = urlparse(url).path
        segments = [s for s in pathname.split('/') if s]
        if segments:
            return segments[0].replace('-', ' ').lower()
        return 'unknown'
    except Exception:
        return 'unknown'

def detect_cannibalizations(df: pd.DataFrame) -> List[Cannibalization]:
    """Detecta canibalizaciones en el dataset"""
    cannibalizations = []
    
    # Agrupar por top_query
    query_groups = df.groupby('top_query')
    
    for query, group in query_groups:
        if len(group) > 1 and pd.notna(query) and str(query).strip():
            # Ordenar por clics
            sorted_group = group.sort_values('top_query_clicks', ascending=False)
            
            urls_data = sorted_group.to_dict('records')
            total_clicks = sorted_group['top_query_clicks'].sum()
            total_impressions = sorted_group['top_query_impressions'].sum()
            
            positions = sorted_group['top_query_position'].dropna()
            position_variance = positions.max() - positions.min() if len(positions) > 0 else 0
            
            # Determinar severidad
            url_count = len(group)
            if url_count > 3:
                severity = Severity.HIGH
            elif url_count > 2:
                severity = Severity.MEDIUM
            else:
                severity = Severity.LOW
            
            cannibalizations.append(Cannibalization(
                query=str(query),
                urls=urls_data,
                url_count=url_count,
                total_clicks=int(total_clicks),
                total_impressions=int(total_impressions),
                position_variance=float(position_variance),
                severity=severity,
                main_url=urls_data[0]['url'],
                competing_urls=[u['url'] for u in urls_data[1:]]
            ))
    
    # Ordenar por severidad y clics
    severity_order = {Severity.HIGH: 3, Severity.MEDIUM: 2, Severity.LOW: 1}
    cannibalizations.sort(key=lambda x: (severity_order[x.severity], x.total_clicks), reverse=True)
    
    return cannibalizations

def generate_recommendations(cannibalization: Cannibalization) -> List[Recommendation]:
    """Genera recomendaciones para una canibalización"""
    recommendations = []
    
    for idx, url_data in enumerate(cannibalization.urls):
        url = url_data['url']
        url_type = classify_url(url)
        is_main = idx == 0
        
        if is_main:
            recommendations.append(Recommendation(
                url=url,
                url_type=url_type,
                action="MANTENER",
                priority="ALTA",
                description="URL principal con mejor rendimiento. Mantener y potenciar.",
                tactics=[
                    "Optimizar meta title y description para la query objetivo",
                    "Reforzar enlazado interno desde otras páginas relevantes",
                    "Añadir contenido complementario si es necesario",
                    "Revisar estructura de encabezados H1-H6"
                ]
            ))
        else:
            click_ratio = url_data.get('top_query_clicks', 0) / max(cannibalization.total_clicks, 1)
            position_diff = url_data.get('top_query_position', 0) - cannibalization.urls[0].get('top_query_position', 0)
            main_type = classify_url(cannibalization.main_url)
            
            if url_type == UrlType.BLOG and main_type == UrlType.PLP:
                recommendations.append(Recommendation(
                    url=url,
                    url_type=url_type,
                    action="REDIRIGIR 301",
                    priority="ALTA",
                    description="Blog compitiendo con PLP comercial. Redirigir o consolidar contenido.",
                    tactics=[
                        "Implementar 301 hacia la PLP principal",
                        "Alternativamente: mover contenido a /blog/ con canonical hacia PLP",
                        "Actualizar anchor texts de enlaces entrantes",
                        "Eliminar de sitemap tras redirección"
                    ]
                ))
            elif url_type == UrlType.PDP and main_type == UrlType.PLP:
                recommendations.append(Recommendation(
                    url=url,
                    url_type=url_type,
                    action="CANONICAL",
                    priority="MEDIA",
                    description="PDP rankeando por query genérica. Añadir canonical o noindex.",
                    tactics=[
                        "Implementar rel='canonical' hacia la PLP padre",
                        "Alternativamente: añadir meta robots noindex",
                        "Revisar contenido duplicado entre PDP y PLP",
                        "Mejorar diferenciación de keywords long-tail en PDP"
                    ]
                ))
            elif click_ratio < 0.1 and position_diff > 10:
                recommendations.append(Recommendation(
                    url=url,
                    url_type=url_type,
                    action="NOINDEX / 410",
                    priority="BAJA",
                    description="URL con bajo rendimiento y posición lejana. Considerar eliminación.",
                    tactics=[
                        "Aplicar meta robots noindex si el contenido tiene valor",
                        "Considerar 410 Gone si el contenido es obsoleto",
                        "Redirigir 301 si hay enlaces entrantes valiosos",
                        "Excluir de crawl vía robots.txt temporalmente"
                    ]
                ))
            else:
                recommendations.append(Recommendation(
                    url=url,
                    url_type=url_type,
                    action="DIFERENCIAR",
                    priority="MEDIA",
                    description="Diferenciar el intent de búsqueda y optimizar para long-tail.",
                    tactics=[
                        "Identificar intent único para esta URL",
                        "Modificar H1, title y contenido hacia keywords específicas",
                        "Añadir enlace contextual hacia URL principal",
                        "Revisar y actualizar internal linking strategy"
                    ]
                ))
    
    return recommendations


# ==================== FUNCIONES DE IA ====================
def analyze_with_anthropic(cannibalization_data: List[Dict], api_key: str) -> str:
    """Analiza canibalizaciones con Claude"""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        
        prompt = f"""Eres un experto SEO técnico especializado en ecommerce.
Analiza los siguientes casos de canibalización de keywords y proporciona:

1. **Resumen Ejecutivo**: Descripción concisa del problema principal
2. **Priorización**: Lista ordenada de URLs a tratar primero  
3. **Acciones Recomendadas**: Para cada caso crítico, especifica la acción y justificación
4. **Impacto Estimado**: Proyección de mejora en tráfico/posiciones
5. **Riesgos**: Posibles efectos negativos a monitorizar

Datos de canibalización:
{json.dumps(cannibalization_data, indent=2, ensure_ascii=False)}

Responde en español y de forma estructurada."""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text
    except ImportError:
        return "❌ Error: La librería 'anthropic' no está instalada. Añádela a requirements.txt"
    except Exception as e:
        return f"❌ Error al conectar con Anthropic: {str(e)}"

def analyze_with_openai(cannibalization_data: List[Dict], api_key: str) -> str:
    """Analiza canibalizaciones con GPT-4"""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            max_tokens=4096,
            messages=[
                {
                    "role": "system",
                    "content": "Eres un consultor SEO senior especializado en ecommerce y arquitectura de información. Responde siempre en español."
                },
                {
                    "role": "user",
                    "content": f"""Analiza estas canibalizaciones SEO y proporciona recomendaciones detalladas:

{json.dumps(cannibalization_data, indent=2, ensure_ascii=False)}

Incluye: resumen ejecutivo, priorización, acciones específicas, impacto estimado y riesgos."""
                }
            ]
        )
        
        return response.choices[0].message.content
    except ImportError:
        return "❌ Error: La librería 'openai' no está instalada. Añádela a requirements.txt"
    except Exception as e:
        return f"❌ Error al conectar con OpenAI: {str(e)}"


# ==================== FUNCIONES DE SEMRUSH ====================
def get_semrush_organic(keyword: str, api_key: str, limit: int = 5) -> pd.DataFrame:
    """Obtiene posiciones orgánicas de Semrush"""
    try:
        url = "https://api.semrush.com/"
        params = {
            "type": "phrase_organic",
            "key": api_key,
            "phrase": keyword,
            "database": "es",
            "display_limit": limit
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text), sep=';')
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error Semrush: {str(e)}")
        return pd.DataFrame()


# ==================== VISUALIZACIONES ====================
def create_internal_linking_graph(df: pd.DataFrame, family: str) -> go.Figure:
    """Crea grafo de enlaces internos propuesto"""
    family_df = df[df['url'].str.lower().str.contains(family.lower(), na=False)].head(15)
    
    if len(family_df) == 0:
        return None
    
    G = nx.DiGraph()
    
    # Clasificar URLs
    plps = []
    pdps = []
    blogs = []
    
    for url in family_df['url'].unique():
        url_type = classify_url(url)
        node_name = urlparse(url).path.split('/')[-1][:20] or 'home'
        
        if url_type == UrlType.PLP:
            plps.append((node_name, url))
        elif url_type == UrlType.PDP:
            pdps.append((node_name, url))
        else:
            blogs.append((node_name, url))
    
    # Añadir nodos
    for name, url in plps:
        G.add_node(name, type='PLP', url=url)
    for name, url in pdps:
        G.add_node(name, type='PDP', url=url)
    for name, url in blogs:
        G.add_node(name, type='BLOG', url=url)
    
    # Crear enlaces propuestos (PLP -> PDPs y Blogs)
    if plps:
        main_plp = plps[0][0]
        for name, _ in pdps[:5]:
            G.add_edge(main_plp, name)
        for name, _ in blogs[:3]:
            G.add_edge(main_plp, name)
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Crear figura
    fig = go.Figure()
    
    # Edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#94a3b8'),
        hoverinfo='none',
        mode='lines'
    ))
    
    # Nodes por tipo - colores accesibles
    colors = {'PLP': '#0284c7', 'PDP': '#7c3aed', 'BLOG': '#059669'}
    
    for node_type, color in colors.items():
        nodes = [n for n in G.nodes() if G.nodes[n].get('type') == node_type]
        if nodes:
            node_x = [pos[n][0] for n in nodes]
            node_y = [pos[n][1] for n in nodes]
            
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(size=40, color=color, line=dict(width=2, color='white')),
                text=nodes,
                textposition="bottom center",
                textfont=dict(color='#1e293b', size=10),
                name=node_type,
                hoverinfo='text',
                hovertext=[f"{n}\n({node_type})" for n in nodes]
            ))
    
    fig.update_layout(
        title=dict(text=f"Propuesta de Enlazado Interno - {family}", font=dict(color='#1e293b', size=16)),
        showlegend=True,
        plot_bgcolor='#f8fafc',
        paper_bgcolor='#ffffff',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(font=dict(color='#1e293b')),
        height=500
    )
    
    return fig

def create_cannibalization_chart(cannibalizations: List[Cannibalization]) -> go.Figure:
    """Crea gráfico de canibalizaciones por severidad"""
    severity_counts = {
        'Alta': len([c for c in cannibalizations if c.severity == Severity.HIGH]),
        'Media': len([c for c in cannibalizations if c.severity == Severity.MEDIUM]),
        'Baja': len([c for c in cannibalizations if c.severity == Severity.LOW])
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(severity_counts.keys()),
            y=list(severity_counts.values()),
            marker_color=['#dc2626', '#d97706', '#16a34a'],
            text=list(severity_counts.values()),
            textposition='auto',
            textfont=dict(color='white', size=14, family='Arial Black')
        )
    ])
    
    fig.update_layout(
        title=dict(text="Canibalizaciones por Severidad", font=dict(color='#1e293b', size=16)),
        xaxis=dict(title="Severidad", color='#1e293b', tickfont=dict(color='#1e293b')),
        yaxis=dict(title="Cantidad", color='#1e293b', tickfont=dict(color='#1e293b')),
        plot_bgcolor='#f8fafc',
        paper_bgcolor='#ffffff',
        height=300
    )
    
    return fig


# ==================== HELPER FUNCTIONS PARA HTML ====================
def get_badge_html(text: str, badge_type: str) -> str:
    """Genera HTML para un badge"""
    return f'<span class="badge badge-{badge_type}">{text}</span>'

def get_url_type_badge(url_type: UrlType) -> str:
    """Genera badge HTML para tipo de URL"""
    type_map = {
        UrlType.PLP: ('PLP', 'plp'),
        UrlType.PDP: ('PDP', 'pdp'),
        UrlType.BLOG: ('BLOG', 'blog'),
        UrlType.OTHER: ('OTHER', 'plp')
    }
    text, css_class = type_map.get(url_type, ('OTHER', 'plp'))
    return get_badge_html(text, css_class)

def get_severity_badge(severity: Severity) -> str:
    """Genera badge HTML para severidad"""
    severity_map = {
        Severity.HIGH: ('Alta', 'high'),
        Severity.MEDIUM: ('Media', 'medium'),
        Severity.LOW: ('Baja', 'low')
    }
    text, css_class = severity_map.get(severity, ('Baja', 'low'))
    return get_badge_html(text, css_class)


# ==================== INTERFAZ PRINCIPAL ====================
def main():
    # Header
    st.title("🎯 SEO Cannibalization Analyzer")
    st.caption("Análisis de canibalizaciones · Arquitectura de información · Crawl budget")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        st.subheader("🔑 APIs de IA")
        anthropic_key = st.text_input("Anthropic API Key", type="password", key="anthropic")
        openai_key = st.text_input("OpenAI API Key", type="password", key="openai")
        semrush_key = st.text_input("Semrush API Key", type="password", key="semrush")
        
        st.divider()
        
        st.subheader("📊 Filtros")
        min_clicks = st.slider("Mín. clics para analizar", 0, 100, 5)
        severity_filter = st.multiselect(
            "Severidad",
            ["Alta", "Media", "Baja"],
            default=["Alta", "Media", "Baja"]
        )
        
        st.divider()
        
        st.warning("⚠️ **Importante**: Validar siempre con el Dpto. SEO antes de implementar cambios.")
    
    # Main content - Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 Cargar Datos", 
        "🔍 Análisis", 
        "💡 Recomendaciones", 
        "🔗 Enlazado Interno", 
        "📈 Competencia"
    ])
    
    # ==================== TAB 1: CARGAR DATOS ====================
    with tab1:
        st.header("Carga tu export de Search Console")
        
        uploaded_file = st.file_uploader(
            "Arrastra tu archivo CSV aquí",
            type=['csv'],
            help="Formato CSV con columnas: url, top_query, top_query_clicks, etc."
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state['df'] = df
                st.success(f"✅ Archivo cargado correctamente: **{len(df):,} filas**")
                
                # Preview
                st.subheader("Vista previa de datos")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Clasificar URLs
                df['url_type'] = df['url'].apply(lambda x: classify_url(x).value)
                df['family'] = df['url'].apply(extract_family_from_url)
                st.session_state['df'] = df
                
                # Métricas generales
                st.subheader("📊 Resumen del dataset")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("URLs totales", f"{len(df):,}")
                with col2:
                    st.metric("Clics totales", f"{df['url_total_clicks'].sum():,}")
                with col3:
                    st.metric("Impresiones", f"{df['url_total_impressions'].sum():,}")
                with col4:
                    avg_pos = df['url_avg_position'].mean()
                    st.metric("Posición media", f"{avg_pos:.1f}")
                
                # Distribución por tipo
                col1, col2 = st.columns(2)
                with col1:
                    type_counts = df['url_type'].value_counts()
                    fig = px.pie(
                        values=type_counts.values,
                        names=type_counts.index,
                        title="Distribución por tipo de URL",
                        color_discrete_sequence=['#0284c7', '#7c3aed', '#059669', '#64748b']
                    )
                    fig.update_layout(
                        plot_bgcolor='#ffffff',
                        paper_bgcolor='#ffffff',
                        font=dict(color='#1e293b')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Top familias
                    family_counts = df['family'].value_counts().head(10)
                    fig = px.bar(
                        x=family_counts.values,
                        y=family_counts.index,
                        orientation='h',
                        title="Top 10 Familias de Productos",
                        color_discrete_sequence=['#0284c7']
                    )
                    fig.update_layout(
                        plot_bgcolor='#f8fafc',
                        paper_bgcolor='#ffffff',
                        font=dict(color='#1e293b'),
                        yaxis=dict(title=""),
                        xaxis=dict(title="Cantidad de URLs")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error al procesar el archivo: {str(e)}")
        else:
            st.info("👆 Sube tu archivo CSV de Search Console para comenzar")
            
            with st.expander("📋 Columnas esperadas en el CSV"):
                st.markdown("""
                | Columna | Descripción |
                |---------|-------------|
                | `url` | URL completa de la página |
                | `top_query` | Query principal que genera más clics |
                | `top_query_clicks` | Clics generados por la top query |
                | `top_query_impressions` | Impresiones de la top query |
                | `top_query_position` | Posición media de la top query |
                | `url_total_clicks` | Clics totales de la URL |
                | `url_total_impressions` | Impresiones totales |
                | `url_avg_position` | Posición media global |
                """)
    
    # ==================== TAB 2: ANÁLISIS ====================
    with tab2:
        if 'df' not in st.session_state:
            st.warning("⚠️ Primero carga un archivo CSV en la pestaña **Cargar Datos**")
        else:
            df = st.session_state['df']
            
            st.header("🔍 Seleccionar familia de productos")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                families = df['family'].value_counts()
                family_options = [f"{f} ({c} URLs)" for f, c in families.head(50).items()]
                selected = st.selectbox("Selecciona una familia", family_options)
                selected_family = selected.split(' (')[0] if selected else None
            
            with col2:
                custom_family = st.text_input("O introduce manualmente")
                if custom_family:
                    selected_family = custom_family.lower()
            
            if selected_family:
                # Filtrar datos
                family_df = df[
                    (df['url'].str.lower().str.contains(selected_family, na=False)) |
                    (df['top_query'].str.lower().str.contains(selected_family, na=False))
                ]
                
                # Detectar canibalizaciones
                cannibalizations = detect_cannibalizations(family_df)
                
                # Filtrar por severidad
                severity_map = {"Alta": Severity.HIGH, "Media": Severity.MEDIUM, "Baja": Severity.LOW}
                filtered_cannib = [
                    c for c in cannibalizations 
                    if c.severity in [severity_map[s] for s in severity_filter]
                    and c.total_clicks >= min_clicks
                ]
                
                st.session_state['cannibalizations'] = filtered_cannib
                st.session_state['selected_family'] = selected_family
                
                # Mostrar resultados
                st.header(f"Canibalizaciones detectadas: `{selected_family}`")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    high = len([c for c in filtered_cannib if c.severity == Severity.HIGH])
                    st.metric("🔴 Alta severidad", high)
                with col2:
                    medium = len([c for c in filtered_cannib if c.severity == Severity.MEDIUM])
                    st.metric("🟡 Media severidad", medium)
                with col3:
                    low = len([c for c in filtered_cannib if c.severity == Severity.LOW])
                    st.metric("🟢 Baja severidad", low)
                
                if filtered_cannib:
                    # Gráfico
                    fig = create_cannibalization_chart(filtered_cannib)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Lista de canibalizaciones
                    st.divider()
                    st.subheader("Detalle de canibalizaciones")
                    
                    for idx, cannib in enumerate(filtered_cannib[:20]):
                        severity_class = {
                            Severity.HIGH: "severity-high",
                            Severity.MEDIUM: "severity-medium",
                            Severity.LOW: "severity-low"
                        }[cannib.severity]
                        
                        severity_emoji = {
                            Severity.HIGH: "🔴",
                            Severity.MEDIUM: "🟡",
                            Severity.LOW: "🟢"
                        }[cannib.severity]
                        
                        with st.expander(f"{severity_emoji} `{cannib.query}` — {cannib.url_count} URLs compitiendo"):
                            # Info general
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Severidad", cannib.severity.value)
                            with col2:
                                st.metric("Clics totales", f"{cannib.total_clicks:,}")
                            with col3:
                                st.metric("Impresiones", f"{cannib.total_impressions:,}")
                            with col4:
                                st.metric("Var. posición", f"{cannib.position_variance:.1f}")
                            
                            st.markdown("**URLs compitiendo:**")
                            
                            for i, url_data in enumerate(cannib.urls):
                                url_type = classify_url(url_data['url'])
                                is_main = i == 0
                                
                                if is_main:
                                    st.markdown(f"""
                                    <div class="url-item-main">
                                        {get_url_type_badge(url_type)} <strong>✅ URL Principal</strong><br>
                                        <a href="{url_data['url']}" target="_blank">{url_data['url'][:80]}...</a><br>
                                        <small>Clics: {url_data.get('top_query_clicks', 0)} | Impresiones: {url_data.get('top_query_impressions', 0):,} | Posición: {url_data.get('top_query_position', 0):.1f}</small>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="url-item">
                                        {get_url_type_badge(url_type)}<br>
                                        <a href="{url_data['url']}" target="_blank">{url_data['url'][:80]}...</a><br>
                                        <small>Clics: {url_data.get('top_query_clicks', 0)} | Impresiones: {url_data.get('top_query_impressions', 0):,} | Posición: {url_data.get('top_query_position', 0):.1f}</small>
                                    </div>
                                    """, unsafe_allow_html=True)
                else:
                    st.success("✅ ¡No se detectaron canibalizaciones significativas para esta familia!")
                
                # Análisis con IA
                st.divider()
                st.subheader("🤖 Análisis con IA")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔮 Analizar con Claude", disabled=not anthropic_key, use_container_width=True):
                        with st.spinner("Analizando con Claude..."):
                            cannib_data = [
                                {"query": c.query, "urls": c.url_count, "clicks": c.total_clicks, "severity": c.severity.value}
                                for c in filtered_cannib[:10]
                            ]
                            result = analyze_with_anthropic(cannib_data, anthropic_key)
                            st.markdown(result)
                
                with col2:
                    if st.button("🧠 Analizar con GPT-4", disabled=not openai_key, use_container_width=True):
                        with st.spinner("Analizando con GPT-4..."):
                            cannib_data = [
                                {"query": c.query, "urls": c.url_count, "clicks": c.total_clicks, "severity": c.severity.value}
                                for c in filtered_cannib[:10]
                            ]
                            result = analyze_with_openai(cannib_data, openai_key)
                            st.markdown(result)
                
                if not anthropic_key and not openai_key:
                    st.info("💡 Introduce una API Key en la barra lateral para habilitar el análisis con IA")
    
    # ==================== TAB 3: RECOMENDACIONES ====================
    with tab3:
        if 'cannibalizations' not in st.session_state or not st.session_state['cannibalizations']:
            st.warning("⚠️ Primero analiza una familia de productos en la pestaña **Análisis**")
        else:
            cannibalizations = st.session_state['cannibalizations']
            
            st.header("💡 Recomendaciones SEO")
            
            st.markdown("""
            <div class="warning-box">
                ⚠️ <strong>Recordatorio:</strong> Todas las acciones deben ser validadas con el Departamento SEO antes de su implementación.
            </div>
            """, unsafe_allow_html=True)
            
            # Generar recomendaciones para las top canibalizaciones
            all_recommendations = []
            for cannib in cannibalizations[:10]:
                recs = generate_recommendations(cannib)
                all_recommendations.extend([(cannib.query, r) for r in recs])
            
            # Separar por tipo de acción
            hard_actions = [r for r in all_recommendations if r[1].action in ['REDIRIGIR 301', 'NOINDEX / 410']]
            soft_actions = [r for r in all_recommendations if r[1].action in ['MANTENER', 'CANONICAL', 'DIFERENCIAR']]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔨 Acciones Hard")
                st.caption("Cambios estructurales (301, noindex, 410)")
                
                if hard_actions:
                    for query, rec in hard_actions:
                        priority_emoji = "🔴" if rec.priority == "ALTA" else "🟡" if rec.priority == "MEDIA" else "🟢"
                        
                        st.markdown(f"""
                        <div class="recommendation-card-hard">
                            <strong>{rec.action}</strong> {get_url_type_badge(rec.url_type)} {priority_emoji} Prioridad {rec.priority}<br>
                            <small><strong>Query:</strong> {query}</small><br>
                            <p>{rec.description}</p>
                            <a href="{rec.url}" target="_blank">{rec.url[:60]}...</a>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("Ver tácticas"):
                            for tactic in rec.tactics:
                                st.markdown(f"- {tactic}")
                else:
                    st.info("No hay acciones hard recomendadas")
            
            with col2:
                st.subheader("🎯 Acciones Soft")
                st.caption("Optimización (canonical, diferenciación)")
                
                if soft_actions:
                    for query, rec in soft_actions[:10]:
                        priority_emoji = "🔴" if rec.priority == "ALTA" else "🟡" if rec.priority == "MEDIA" else "🟢"
                        
                        st.markdown(f"""
                        <div class="recommendation-card-soft">
                            <strong>{rec.action}</strong> {get_url_type_badge(rec.url_type)} {priority_emoji} Prioridad {rec.priority}<br>
                            <small><strong>Query:</strong> {query}</small><br>
                            <p>{rec.description}</p>
                            <a href="{rec.url}" target="_blank">{rec.url[:60]}...</a>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("Ver tácticas"):
                            for tactic in rec.tactics:
                                st.markdown(f"- {tactic}")
                else:
                    st.info("No hay acciones soft recomendadas")
            
            # Exportar recomendaciones
            st.divider()
            st.subheader("📥 Exportar recomendaciones")
            
            if st.button("Generar CSV de recomendaciones", use_container_width=True):
                export_data = []
                for query, rec in all_recommendations:
                    export_data.append({
                        'query': query,
                        'url': rec.url,
                        'tipo_url': rec.url_type.value,
                        'accion': rec.action,
                        'prioridad': rec.priority,
                        'descripcion': rec.description,
                        'tacticas': ' | '.join(rec.tactics)
                    })
                
                export_df = pd.DataFrame(export_data)
                csv = export_df.to_csv(index=False)
                st.download_button(
                    "💾 Descargar CSV",
                    csv,
                    "recomendaciones_seo.csv",
                    "text/csv",
                    use_container_width=True
                )
    
    # ==================== TAB 4: ENLAZADO INTERNO ====================
    with tab4:
        if 'selected_family' not in st.session_state or 'df' not in st.session_state:
            st.warning("⚠️ Primero selecciona una familia de productos en la pestaña **Análisis**")
        else:
            family = st.session_state['selected_family']
            df = st.session_state['df']
            
            st.header(f"🔗 Arquitectura de enlaces internos")
            st.caption(f"Familia: **{family}**")
            
            # Gráfico
            fig = create_internal_linking_graph(df, family)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay suficientes URLs para generar el gráfico")
            
            # Recomendaciones de enlazado
            st.subheader("📋 Recomendaciones de enlazado interno")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="recommendation-card-soft">
                    <h4>✅ Buenas prácticas</h4>
                    <ul>
                        <li>Enlazar desde PLP principal hacia todos los PDPs hijos</li>
                        <li>Usar anchor text descriptivo y con keywords</li>
                        <li>Implementar breadcrumbs semánticos</li>
                        <li>Posts del blog deben enlazar a PLPs (no PDPs)</li>
                        <li>Máximo 3 niveles de profundidad</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="recommendation-card-hard">
                    <h4>❌ Evitar</h4>
                    <ul>
                        <li>Enlaces cruzados entre PDPs del mismo nivel</li>
                        <li>Orphan pages (páginas sin enlaces entrantes)</li>
                        <li>Exceso de enlaces en una misma página (+100)</li>
                        <li>Enlaces con anchor text genérico ("click aquí")</li>
                        <li>Cadenas de redirecciones</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Estructura propuesta
            st.subheader("🏗️ Estructura jerárquica propuesta")
            
            family_df = df[df['url'].str.lower().str.contains(family, na=False)]
            
            plps = family_df[family_df['url_type'] == 'PLP']['url'].tolist()[:5]
            pdps = family_df[family_df['url_type'] == 'PDP']['url'].tolist()[:10]
            blogs = family_df[family_df['url_type'] == 'BLOG']['url'].tolist()[:3]
            
            st.code(f"""
📁 {family.upper()} (PLP Principal)
│
├── 📄 PDPs ({len(pdps)} productos)
│   ├── {plps[0] if plps else 'N/A'}
│   ├── {pdps[1] if len(pdps) > 1 else '...'}
│   └── ... ({len(pdps)} más)
│
└── 📝 Blog/Guías ({len(blogs)} artículos)
    ├── {blogs[0] if blogs else 'N/A'}
    └── ... ({len(blogs)} más)
            """, language=None)
    
    # ==================== TAB 5: COMPETENCIA ====================
    with tab5:
        if 'selected_family' not in st.session_state:
            st.warning("⚠️ Primero selecciona una familia de productos en la pestaña **Análisis**")
        else:
            family = st.session_state['selected_family']
            
            st.header(f"📈 Análisis de competencia")
            st.caption(f"Familia: **{family}**")
            
            if not semrush_key:
                st.info("🔑 Introduce tu API Key de Semrush en la barra lateral para activar el análisis de competencia")
            else:
                keyword = st.text_input("Keyword a analizar", value=family)
                
                if st.button("🔍 Analizar Top 5 Orgánico", use_container_width=True):
                    with st.spinner("Consultando Semrush..."):
                        results = get_semrush_organic(keyword, semrush_key, limit=5)
                        
                        if not results.empty:
                            st.subheader("🏆 Top 5 Posiciones Orgánicas")
                            
                            for idx, row in results.iterrows():
                                position = idx + 1
                                medal = "🥇" if position == 1 else "🥈" if position == 2 else "🥉" if position == 3 else f"#{position}"
                                
                                st.markdown(f"""
                                <div class="recommendation-card">
                                    <strong>{medal} Posición {position}</strong><br>
                                    <strong>{row.get('Domain', 'N/A')}</strong><br>
                                    <a href="{row.get('Url', '#')}" target="_blank">{row.get('Url', 'N/A')[:60]}...</a><br>
                                    <small>Traffic: {row.get('Traffic', 'N/A')} | CPC: {row.get('CPC', 'N/A')}</small>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.warning("No se encontraron resultados para esta keyword")
            
            # Análisis manual de competencia
            st.divider()
            st.subheader("🔬 Análisis manual de competidores")
            
            competitor_url = st.text_input("URL de competidor a analizar")
            
            if competitor_url and st.button("Analizar estructura"):
                try:
                    parsed = urlparse(competitor_url)
                    path_segments = [s for s in parsed.path.split('/') if s]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Dominio", parsed.netloc)
                    with col2:
                        st.metric("Profundidad", f"{len(path_segments)} niveles")
                    with col3:
                        st.metric("Tipo estimado", classify_url(competitor_url).value)
                    
                    st.code(f"Estructura: /{'/'.join(path_segments)}", language=None)
                    
                    if len(path_segments) <= 2:
                        st.success("✅ Estructura plana - Buena para SEO")
                    elif len(path_segments) <= 4:
                        st.warning("⚠️ Estructura moderada - Aceptable")
                    else:
                        st.error("❌ Estructura profunda - Puede afectar crawl budget")
                        
                except Exception as e:
                    st.error(f"Error al analizar URL: {str(e)}")


if __name__ == "__main__":
    main()
