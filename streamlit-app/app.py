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
import anthropic
import openai
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

# CSS personalizado
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .main { padding: 0rem 1rem; }
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%); }
    
    h1, h2, h3 { color: #f1f5f9 !important; font-family: 'Inter', sans-serif; }
    p, span, label { color: #cbd5e1 !important; }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(71, 85, 105, 0.5);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    
    .severity-high { 
        background: rgba(239, 68, 68, 0.1); 
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .severity-medium { 
        background: rgba(245, 158, 11, 0.1); 
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .severity-low { 
        background: rgba(16, 185, 129, 0.1); 
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .recommendation-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(71, 85, 105, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .url-badge-plp { 
        background: #06b6d4; 
        color: white; 
        padding: 2px 8px; 
        border-radius: 4px; 
        font-size: 12px;
        font-weight: 600;
    }
    .url-badge-pdp { 
        background: #8b5cf6; 
        color: white; 
        padding: 2px 8px; 
        border-radius: 4px; 
        font-size: 12px;
        font-weight: 600;
    }
    .url-badge-blog { 
        background: #10b981; 
        color: white; 
        padding: 2px 8px; 
        border-radius: 4px; 
        font-size: 12px;
        font-weight: 600;
    }
    
    .warning-box {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    div[data-testid="stMetricValue"] { color: #22d3ee !important; font-size: 2rem !important; }
    div[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 8px;
        padding: 8px 16px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0891b2, #2563eb);
        color: white !important;
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
                    action="REDIRIGIR_301",
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
                    action="NOINDEX_O_410",
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
    except Exception as e:
        return f"Error al conectar con Anthropic: {str(e)}"

def analyze_with_openai(cannibalization_data: List[Dict], api_key: str) -> str:
    """Analiza canibalizaciones con GPT-4"""
    try:
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
    except Exception as e:
        return f"Error al conectar con OpenAI: {str(e)}"


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
        line=dict(width=1, color='#64748b'),
        hoverinfo='none',
        mode='lines'
    ))
    
    # Nodes por tipo
    colors = {'PLP': '#06b6d4', 'PDP': '#8b5cf6', 'BLOG': '#10b981'}
    
    for node_type, color in colors.items():
        nodes = [n for n in G.nodes() if G.nodes[n].get('type') == node_type]
        if nodes:
            node_x = [pos[n][0] for n in nodes]
            node_y = [pos[n][1] for n in nodes]
            
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(size=30, color=color, line=dict(width=2, color='white')),
                text=nodes,
                textposition="bottom center",
                textfont=dict(color='white', size=10),
                name=node_type,
                hoverinfo='text',
                hovertext=[f"{n}\n({node_type})" for n in nodes]
            ))
    
    fig.update_layout(
        title=dict(text=f"Propuesta de Enlazado Interno - {family}", font=dict(color='white')),
        showlegend=True,
        plot_bgcolor='rgba(15, 23, 42, 0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(font=dict(color='white')),
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
            marker_color=['#ef4444', '#f59e0b', '#10b981'],
            text=list(severity_counts.values()),
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=dict(text="Canibalizaciones por Severidad", font=dict(color='white')),
        xaxis=dict(title="Severidad", color='white'),
        yaxis=dict(title="Cantidad", color='white'),
        plot_bgcolor='rgba(15, 23, 42, 0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=300
    )
    
    return fig


# ==================== INTERFAZ PRINCIPAL ====================
def main():
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🎯 SEO Cannibalization Analyzer")
        st.caption("Análisis de canibalizaciones · Arquitectura de información · Crawl budget")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/seo.png", width=80)
        st.markdown("### ⚙️ Configuración")
        
        st.markdown("---")
        st.markdown("#### 🔑 APIs de IA")
        
        anthropic_key = st.text_input("Anthropic API Key", type="password", key="anthropic")
        openai_key = st.text_input("OpenAI API Key", type="password", key="openai")
        semrush_key = st.text_input("Semrush API Key", type="password", key="semrush")
        
        st.markdown("---")
        st.markdown("#### 📊 Filtros")
        
        min_clicks = st.slider("Mín. clics para analizar", 0, 100, 5)
        severity_filter = st.multiselect(
            "Severidad",
            ["Alta", "Media", "Baja"],
            default=["Alta", "Media", "Baja"]
        )
        
        st.markdown("---")
        st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>Importante</strong><br>
            Validar siempre con el Dpto. SEO antes de implementar cambios.
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    tabs = st.tabs(["📤 Cargar Datos", "🔍 Análisis", "💡 Recomendaciones", "🔗 Enlazado Interno", "📈 Competencia"])
    
    # ==================== TAB 1: CARGAR DATOS ====================
    with tabs[0]:
        st.markdown("### Carga tu export de Search Console")
        
        uploaded_file = st.file_uploader(
            "Arrastra tu archivo CSV aquí",
            type=['csv'],
            help="Formato CSV con columnas: url, top_query, top_query_clicks, etc."
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state['df'] = df
                st.success(f"✅ Archivo cargado: {len(df):,} filas")
                
                # Preview
                st.markdown("#### Vista previa")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Clasificar URLs
                df['url_type'] = df['url'].apply(lambda x: classify_url(x).value)
                df['family'] = df['url'].apply(extract_family_from_url)
                st.session_state['df'] = df
                
                # Métricas generales
                st.markdown("#### 📊 Resumen del dataset")
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
                        color_discrete_sequence=['#06b6d4', '#8b5cf6', '#10b981', '#64748b']
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
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
                        color_discrete_sequence=['#06b6d4']
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(15, 23, 42, 0.8)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        yaxis=dict(title=""),
                        xaxis=dict(title="Cantidad de URLs")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error al procesar el archivo: {str(e)}")
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
    with tabs[1]:
        if 'df' not in st.session_state:
            st.warning("⚠️ Primero carga un archivo CSV en la pestaña 'Cargar Datos'")
        else:
            df = st.session_state['df']
            
            st.markdown("### 🔍 Seleccionar familia de productos")
            
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
                st.markdown(f"### Canibalizaciones detectadas: `{selected_family}`")
                
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
                    st.markdown("---")
                    for idx, cannib in enumerate(filtered_cannib[:20]):
                        severity_class = {
                            Severity.HIGH: "severity-high",
                            Severity.MEDIUM: "severity-medium",
                            Severity.LOW: "severity-low"
                        }[cannib.severity]
                        
                        with st.expander(f"🔍 `{cannib.query}` - {cannib.url_count} URLs compitiendo"):
                            st.markdown(f"""
                            <div class="{severity_class}">
                                <strong>Severidad:</strong> {cannib.severity.value}<br>
                                <strong>Clics totales:</strong> {cannib.total_clicks:,}<br>
                                <strong>Impresiones:</strong> {cannib.total_impressions:,}<br>
                                <strong>Varianza posición:</strong> {cannib.position_variance:.1f}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("#### URLs compitiendo:")
                            for i, url_data in enumerate(cannib.urls):
                                url_type = classify_url(url_data['url'])
                                badge_class = f"url-badge-{url_type.value.lower()}"
                                is_main = "✅ Principal" if i == 0 else ""
                                
                                st.markdown(f"""
                                <div style="padding: 0.5rem; margin: 0.25rem 0; background: rgba(30,41,59,0.5); border-radius: 6px;">
                                    <span class="{badge_class}">{url_type.value}</span>
                                    <strong>{is_main}</strong><br>
                                    <a href="{url_data['url']}" target="_blank" style="color: #22d3ee;">{url_data['url'][:80]}...</a><br>
                                    <small>Clics: {url_data.get('top_query_clicks', 0)} | Impresiones: {url_data.get('top_query_impressions', 0):,} | Pos: {url_data.get('top_query_position', 0):.1f}</small>
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    st.success("✅ ¡No se detectaron canibalizaciones significativas para esta familia!")
                
                # Análisis con IA
                st.markdown("---")
                st.markdown("### 🤖 Análisis con IA")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔮 Analizar con Claude", disabled=not anthropic_key):
                        with st.spinner("Analizando con Claude..."):
                            cannib_data = [
                                {"query": c.query, "urls": c.url_count, "clicks": c.total_clicks, "severity": c.severity.value}
                                for c in filtered_cannib[:10]
                            ]
                            result = analyze_with_anthropic(cannib_data, anthropic_key)
                            st.markdown(result)
                
                with col2:
                    if st.button("🧠 Analizar con GPT-4", disabled=not openai_key):
                        with st.spinner("Analizando con GPT-4..."):
                            cannib_data = [
                                {"query": c.query, "urls": c.url_count, "clicks": c.total_clicks, "severity": c.severity.value}
                                for c in filtered_cannib[:10]
                            ]
                            result = analyze_with_openai(cannib_data, openai_key)
                            st.markdown(result)
    
    # ==================== TAB 3: RECOMENDACIONES ====================
    with tabs[2]:
        if 'cannibalizations' not in st.session_state or not st.session_state['cannibalizations']:
            st.warning("⚠️ Primero analiza una familia de productos en la pestaña 'Análisis'")
        else:
            cannibalizations = st.session_state['cannibalizations']
            
            st.markdown("### 💡 Recomendaciones SEO")
            
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
            hard_actions = [r for r in all_recommendations if r[1].action in ['REDIRIGIR_301', 'NOINDEX_O_410']]
            soft_actions = [r for r in all_recommendations if r[1].action in ['MANTENER', 'CANONICAL', 'DIFERENCIAR']]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔨 Acciones Hard (cambios estructurales)")
                for query, rec in hard_actions:
                    action_colors = {
                        'REDIRIGIR_301': '#ef4444',
                        'NOINDEX_O_410': '#8b5cf6'
                    }
                    color = action_colors.get(rec.action, '#64748b')
                    
                    st.markdown(f"""
                    <div class="recommendation-card" style="border-left: 4px solid {color};">
                        <strong style="color: {color};">{rec.action.replace('_', ' ')}</strong>
                        <span class="url-badge-{rec.url_type.value.lower()}">{rec.url_type.value}</span>
                        <span style="background: {'#ef4444' if rec.priority == 'ALTA' else '#f59e0b'}; color: white; padding: 2px 6px; border-radius: 4px; font-size: 11px;">
                            Prioridad {rec.priority}
                        </span>
                        <p style="margin: 0.5rem 0; font-size: 14px;">{rec.description}</p>
                        <a href="{rec.url}" target="_blank" style="color: #22d3ee; font-size: 12px; word-break: break-all;">{rec.url[:60]}...</a>
                        <details>
                            <summary style="cursor: pointer; margin-top: 0.5rem;">Ver tácticas</summary>
                            <ul style="margin-top: 0.5rem;">
                                {"".join(f"<li style='font-size: 13px;'>{t}</li>" for t in rec.tactics)}
                            </ul>
                        </details>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 🎯 Acciones Soft (optimización)")
                for query, rec in soft_actions[:10]:
                    action_colors = {
                        'MANTENER': '#10b981',
                        'CANONICAL': '#f59e0b',
                        'DIFERENCIAR': '#06b6d4'
                    }
                    color = action_colors.get(rec.action, '#64748b')
                    
                    st.markdown(f"""
                    <div class="recommendation-card" style="border-left: 4px solid {color};">
                        <strong style="color: {color};">{rec.action.replace('_', ' ')}</strong>
                        <span class="url-badge-{rec.url_type.value.lower()}">{rec.url_type.value}</span>
                        <span style="background: {'#ef4444' if rec.priority == 'ALTA' else '#f59e0b' if rec.priority == 'MEDIA' else '#10b981'}; color: white; padding: 2px 6px; border-radius: 4px; font-size: 11px;">
                            Prioridad {rec.priority}
                        </span>
                        <p style="margin: 0.5rem 0; font-size: 14px;">{rec.description}</p>
                        <a href="{rec.url}" target="_blank" style="color: #22d3ee; font-size: 12px; word-break: break-all;">{rec.url[:60]}...</a>
                        <details>
                            <summary style="cursor: pointer; margin-top: 0.5rem;">Ver tácticas</summary>
                            <ul style="margin-top: 0.5rem;">
                                {"".join(f"<li style='font-size: 13px;'>{t}</li>" for t in rec.tactics)}
                            </ul>
                        </details>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Exportar recomendaciones
            st.markdown("---")
            if st.button("📥 Exportar recomendaciones a CSV"):
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
                    "text/csv"
                )
    
    # ==================== TAB 4: ENLAZADO INTERNO ====================
    with tabs[3]:
        if 'selected_family' not in st.session_state or 'df' not in st.session_state:
            st.warning("⚠️ Primero selecciona una familia de productos en la pestaña 'Análisis'")
        else:
            family = st.session_state['selected_family']
            df = st.session_state['df']
            
            st.markdown(f"### 🔗 Arquitectura de enlaces internos - `{family}`")
            
            # Gráfico
            fig = create_internal_linking_graph(df, family)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Recomendaciones de enlazado
            st.markdown("#### 📋 Recomendaciones de enlazado interno")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="recommendation-card">
                    <h4 style="color: #10b981;">✅ Buenas prácticas</h4>
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
                <div class="recommendation-card">
                    <h4 style="color: #ef4444;">❌ Evitar</h4>
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
            st.markdown("#### 🏗️ Estructura jerárquica propuesta")
            
            family_df = df[df['url'].str.lower().str.contains(family, na=False)]
            
            plps = family_df[family_df['url_type'] == 'PLP']['url'].tolist()[:5]
            pdps = family_df[family_df['url_type'] == 'PDP']['url'].tolist()[:10]
            blogs = family_df[family_df['url_type'] == 'BLOG']['url'].tolist()[:3]
            
            st.markdown(f"""
            ```
            📁 {family.upper()} (PLP Principal)
            │
            ├── 📄 PDPs ({len(pdps)} productos)
            │   ├── {pdps[0] if pdps else 'N/A'}
            │   ├── {pdps[1] if len(pdps) > 1 else '...'}
            │   └── ... ({len(pdps)} más)
            │
            └── 📝 Blog/Guías ({len(blogs)} artículos)
                ├── {blogs[0] if blogs else 'N/A'}
                └── ... ({len(blogs)} más)
            ```
            """)
    
    # ==================== TAB 5: COMPETENCIA ====================
    with tabs[4]:
        if 'selected_family' not in st.session_state:
            st.warning("⚠️ Primero selecciona una familia de productos en la pestaña 'Análisis'")
        else:
            family = st.session_state['selected_family']
            
            st.markdown(f"### 📈 Análisis de competencia - `{family}`")
            
            if not semrush_key:
                st.info("🔑 Introduce tu API Key de Semrush en la barra lateral para activar el análisis de competencia")
            else:
                keyword = st.text_input("Keyword a analizar", value=family)
                
                if st.button("🔍 Analizar Top 5 Orgánico"):
                    with st.spinner("Consultando Semrush..."):
                        results = get_semrush_organic(keyword, semrush_key, limit=5)
                        
                        if not results.empty:
                            st.markdown("#### 🏆 Top 5 Posiciones Orgánicas")
                            
                            for idx, row in results.iterrows():
                                position = idx + 1
                                medal = "🥇" if position == 1 else "🥈" if position == 2 else "🥉" if position == 3 else "🏅"
                                
                                st.markdown(f"""
                                <div class="recommendation-card">
                                    <span style="font-size: 24px;">{medal}</span>
                                    <strong>Posición {position}</strong><br>
                                    <span style="color: #22d3ee;">{row.get('Domain', 'N/A')}</span><br>
                                    <a href="{row.get('Url', '#')}" target="_blank" style="color: #94a3b8; font-size: 12px;">
                                        {row.get('Url', 'N/A')[:60]}...
                                    </a><br>
                                    <small>Traffic: {row.get('Traffic', 'N/A')} | CPC: {row.get('CPC', 'N/A')}</small>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.warning("No se encontraron resultados para esta keyword")
            
            # Análisis manual de competencia
            st.markdown("---")
            st.markdown("#### 🔬 Análisis manual de competidores")
            
            competitor_url = st.text_input("URL de competidor a analizar")
            
            if competitor_url and st.button("Analizar estructura"):
                try:
                    parsed = urlparse(competitor_url)
                    path_segments = [s for s in parsed.path.split('/') if s]
                    
                    st.markdown(f"""
                    **Dominio:** `{parsed.netloc}`  
                    **Profundidad:** {len(path_segments)} niveles  
                    **Estructura:** `/{'/'.join(path_segments)}`
                    """)
                    
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
