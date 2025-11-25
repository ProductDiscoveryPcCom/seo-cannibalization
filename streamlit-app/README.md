# 🎯 SEO Cannibalization Analyzer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

Herramienta de análisis de canibalizaciones SEO diseñada para ecommerce, que utiliza datos de Google Search Console para detectar, analizar y proponer soluciones a problemas de canibalización de keywords y arquitectura de información deficiente.

![Screenshot](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🚀 Características

### 📊 Análisis de Canibalizaciones
- Detección automática de queries con múltiples URLs compitiendo
- Clasificación por severidad (Alta, Media, Baja)
- Métricas detalladas: clics, impresiones, posición, CTR
- Filtros personalizables

### 💡 Recomendaciones SEO Inteligentes
- **301 Redirect** → Blog compitiendo con PLP comercial
- **Canonical** → PDP rankeando por query genérica  
- **Noindex/410** → URLs de bajo rendimiento
- **Diferenciar** → Optimizar para long-tail

### 🔗 Arquitectura de Enlaces
- Visualización de grafo de enlaces internos
- Propuestas de enlazado optimizado
- Detección de páginas huérfanas

### 🤖 Análisis con IA
- Integración con **Anthropic Claude**
- Integración con **OpenAI GPT-4**
- Insights automatizados y priorización

### 📈 Análisis de Competencia
- Integración con **Semrush API**
- Top 5 posiciones orgánicas
- Análisis de estructura de competidores

## 📦 Instalación Local

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/seo-cannibalization-analyzer.git
cd seo-cannibalization-analyzer

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar
streamlit run app.py
```

## ☁️ Deploy en Streamlit Cloud

1. **Fork** este repositorio a tu cuenta de GitHub

2. Ve a [share.streamlit.io](https://share.streamlit.io)

3. Conecta tu cuenta de GitHub

4. Selecciona el repositorio y rama `main`

5. Configura las **Secrets** (opcional):
   ```toml
   ANTHROPIC_API_KEY = "tu-api-key"
   OPENAI_API_KEY = "tu-api-key"
   SEMRUSH_API_KEY = "tu-api-key"
   ```

6. ¡Deploy! 🚀

## 📋 Formato del CSV

El archivo CSV debe contener las siguientes columnas:

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `url` | string | URL completa de la página |
| `top_query` | string | Query principal que genera más clics |
| `top_query_clicks` | int | Clics generados por la top query |
| `top_query_impressions` | int | Impresiones de la top query |
| `top_query_position` | float | Posición media de la top query |
| `url_total_clicks` | int | Clics totales de la URL |
| `url_total_impressions` | int | Impresiones totales |
| `url_avg_position` | float | Posición media global |

### Ejemplo:
```csv
url,top_query,top_query_clicks,top_query_impressions,top_query_position,url_total_clicks,url_total_impressions,url_avg_position
https://example.com/portatiles,portatiles gaming,234,5600,3.2,515,12000,4.1
https://example.com/portatiles/gaming,portatiles gaming,189,4200,4.1,320,8500,5.2
```

## 🔧 Clasificación de URLs

La herramienta clasifica automáticamente las URLs en tres tipos:

| Tipo | Descripción | Ejemplo |
|------|-------------|---------|
| **PLP** | Product Listing Page | `/portatiles/gaming/` |
| **PDP** | Product Detail Page | `/portatil-asus-rog-strix-g15-ryzen-9` |
| **BLOG** | Posts informativos | `/blog/mejores-portatiles-2025` |

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Añade nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## ⚠️ Disclaimer

**IMPORTANTE**: Todas las recomendaciones generadas por esta herramienta deben ser validadas con el Departamento SEO antes de su implementación. Los cambios estructurales (301, 410, noindex) pueden tener impacto significativo en el posicionamiento.

## 📄 Licencia

MIT License - ver [LICENSE](LICENSE) para más detalles.

## 🙏 Créditos

Desarrollado para análisis SEO técnico en ecommerce.

---

**¿Problemas o sugerencias?** Abre un [issue](https://github.com/tu-usuario/seo-cannibalization-analyzer/issues) en GitHub.
