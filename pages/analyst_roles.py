import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Analyst Roles Guide",
    page_icon="👨‍💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS from main app
try:
    with open('assets/styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except:
    pass

# Page title
st.markdown('<h1 class="main-header">Data & Business Analyst Roles</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Comprehensive guide to the roles, responsibilities and workflow of Analysts</p>', unsafe_allow_html=True)

# Main content
st.markdown('''
<div class="dashboard-card" style="padding: 25px; margin-bottom: 30px;">
<h2 style="color:#3366FF; border-bottom: 2px solid #3366FF; padding-bottom: 10px;">Data Analyst vs. Business Analyst: Understanding the Differences</h2>

<p>While both Data Analysts and Business Analysts work with data to help organizations make better decisions, 
their roles, focus areas, and day-to-day responsibilities differ in significant ways.</p>
</div>
''', unsafe_allow_html=True)

# Create two columns for comparison
col1, col2 = st.columns(2)

with col1:
    st.markdown('''
    <div class="dashboard-card" style="height: 100%; padding: 20px;">
    <h2 style="color:#3366FF; text-align: center;">Data Analyst Role</h2>
    
    <h3 style="color:#1E2A78;">Primary Focus</h3>
    <p>Data Analysts focus on collecting, processing, and performing statistical analyses on large datasets. They are primarily concerned with interpreting numerical data and using it to identify trends, create visualizations, and develop insights.</p>
    
    <h3 style="color:#1E2A78;">Key Responsibilities</h3>
    <ul>
        <li>Collecting and cleaning large sets of structured and unstructured data</li>
        <li>Developing and maintaining databases and data systems</li>
        <li>Using statistical tools to interpret data sets</li>
        <li>Creating data visualizations to present findings</li>
        <li>Developing and implementing data analyses and data collection systems</li>
        <li>Identifying patterns and trends in complex data sets</li>
        <li>Working with programming languages like Python, R, and SQL</li>
        <li>Collaborating with management to prioritize business and information needs</li>
    </ul>
    
    <h3 style="color:#1E2A78;">Technical Skills</h3>
    <ul>
        <li>Strong statistical analysis and math skills</li>
        <li>Proficiency in database languages like SQL</li>
        <li>Experience with data visualization tools (Tableau, Power BI, etc.)</li>
        <li>Programming experience (Python, R, etc.)</li>
        <li>Understanding of data warehousing and ETL processes</li>
        <li>Knowledge of machine learning techniques</li>
        <li>Familiarity with big data processing frameworks</li>
    </ul>
    
    <h3 style="color:#1E2A78;">Workflow & Methodologies</h3>
    <ol>
        <li><strong>Data Collection</strong> - Gathering raw data from various sources</li>
        <li><strong>Data Cleaning</strong> - Processing and cleaning data for accuracy</li>
        <li><strong>Exploratory Analysis</strong> - Initial data investigation to discover patterns</li>
        <li><strong>Statistical Modeling</strong> - Applying statistical methods to analyze data</li>
        <li><strong>Data Visualization</strong> - Creating visual representations of findings</li>
        <li><strong>Reporting</strong> - Communicating insights to stakeholders</li>
        <li><strong>Iterative Analysis</strong> - Refining findings based on feedback</li>
    </ol>
    </div>
    ''', unsafe_allow_html=True)

with col2:
    st.markdown('''
    <div class="dashboard-card" style="height: 100%; padding: 20px;">
    <h2 style="color:#3366FF; text-align: center;">Business Analyst Role</h2>
    
    <h3 style="color:#1E2A78;">Primary Focus</h3>
    <p>Business Analysts bridge the gap between IT and business using data analytics to assess processes, determine requirements, and deliver data-driven recommendations and reports to executives and stakeholders. They focus on improving business operations and processes.</p>
    
    <h3 style="color:#1E2A78;">Key Responsibilities</h3>
    <ul>
        <li>Analyzing business processes and identifying areas for improvement</li>
        <li>Gathering and documenting business requirements</li>
        <li>Translating business needs into technical requirements</li>
        <li>Creating detailed business analysis, outlining problems, opportunities, and solutions</li>
        <li>Conducting cost-benefit analysis and defining ROI for business initiatives</li>
        <li>Working closely with stakeholders to ensure solutions meet business needs</li>
        <li>Creating functional specifications for system development</li>
        <li>Conducting user acceptance testing</li>
    </ul>
    
    <h3 style="color:#1E2A78;">Technical Skills</h3>
    <ul>
        <li>Business process modeling and documentation</li>
        <li>Requirements gathering and management</li>
        <li>Data analysis and interpretation</li>
        <li>Project management</li>
        <li>SQL and database knowledge (basic to intermediate)</li>
        <li>Understanding of business intelligence tools</li>
        <li>Knowledge of enterprise resource planning systems</li>
    </ul>
    
    <h3 style="color:#1E2A78;">Workflow & Methodologies</h3>
    <ol>
        <li><strong>Business Problem Analysis</strong> - Understanding business challenges</li>
        <li><strong>Requirements Gathering</strong> - Collecting stakeholder needs and objectives</li>
        <li><strong>Process Mapping</strong> - Documenting current and future state processes</li>
        <li><strong>Solution Design</strong> - Creating functional specifications for solutions</li>
        <li><strong>Implementation Support</strong> - Assisting in deploying solutions</li>
        <li><strong>User Acceptance Testing</strong> - Ensuring solutions meet requirements</li>
        <li><strong>Business Value Assessment</strong> - Measuring impact and ROI</li>
    </ol>
    </div>
    ''', unsafe_allow_html=True)

# Combined Workflow Section
st.markdown('''
<div class="dashboard-card" style="padding: 25px; margin-top: 30px;">
<h2 style="color:#3366FF; border-bottom: 2px solid #3366FF; padding-bottom: 10px;">How Analysts Work with This Dashboard</h2>

<p>This Business Analytics Dashboard is designed to support both Data Analysts and Business Analysts throughout their workflow. Here's how each role typically uses this tool:</p>

<div style="display: flex; margin-top: 20px;">
    <div style="flex: 1; padding-right: 15px;">
        <h3 style="color:#1E2A78;">Data Analyst Workflow</h3>
        <ol>
            <li><strong>Data Import & Inspection</strong> - Upload CSV data and examine data quality</li>
            <li><strong>Data Cleaning</strong> - Use the cleaning options to handle missing values and outliers</li>
            <li><strong>Exploratory Analysis</strong> - Utilize visualization tools to identify patterns and trends</li>
            <li><strong>Statistical Analysis</strong> - Perform correlation analysis and statistical measurements</li>
            <li><strong>Creating Visualizations</strong> - Build custom charts to illustrate findings</li>
            <li><strong>Generating Insights</strong> - Document key observations and statistical findings</li>
            <li><strong>Saving Analysis</strong> - Store datasets, visualizations, and insights in the database</li>
        </ol>
    </div>
    
    <div style="flex: 1; padding-left: 15px;">
        <h3 style="color:#1E2A78;">Business Analyst Workflow</h3>
        <ol>
            <li><strong>Business Problem Definition</strong> - Articulate the business questions to be answered</li>
            <li><strong>KPI Identification</strong> - Determine which metrics matter for business success</li>
            <li><strong>Data Exploration</strong> - Use dashboard visualizations to explore relevant business data</li>
            <li><strong>Trend Analysis</strong> - Identify patterns relevant to business operations</li>
            <li><strong>Insight Development</strong> - Translate data findings into business implications</li>
            <li><strong>Report Generation</strong> - Create professional PDF reports with business recommendations</li>
            <li><strong>Stakeholder Communication</strong> - Share findings with business stakeholders</li>
        </ol>
    </div>
</div>
</div>
''', unsafe_allow_html=True)

# Skills and Certifications Section
st.markdown('''
<div class="dashboard-card" style="padding: 25px; margin-top: 30px;">
<h2 style="color:#3366FF; border-bottom: 2px solid #3366FF; padding-bottom: 10px;">Skills Development & Certifications</h2>

<p>To excel in analyst roles, continuous skill development is essential. Here are recommended skills and certifications for each role:</p>

<div style="display: flex; margin-top: 20px;">
    <div style="flex: 1; padding-right: 15px;">
        <h3 style="color:#1E2A78;">Data Analyst Path</h3>
        <h4>Core Skills to Develop:</h4>
        <ul>
            <li>Advanced SQL and Database Management</li>
            <li>Programming in Python or R</li>
            <li>Statistical Analysis and Hypothesis Testing</li>
            <li>Data Visualization (Tableau, Power BI)</li>
            <li>Machine Learning Fundamentals</li>
            <li>Big Data Technologies</li>
        </ul>
        
        <h4>Recommended Certifications:</h4>
        <ul>
            <li>Microsoft Certified: Data Analyst Associate</li>
            <li>Tableau Desktop Specialist</li>
            <li>IBM Data Analyst Professional Certificate</li>
            <li>Google Data Analytics Professional Certificate</li>
            <li>SAS Certified Data Scientist</li>
        </ul>
    </div>
    
    <div style="flex: 1; padding-left: 15px;">
        <h3 style="color:#1E2A78;">Business Analyst Path</h3>
        <h4>Core Skills to Develop:</h4>
        <ul>
            <li>Business Process Modeling</li>
            <li>Requirements Engineering</li>
            <li>Agile Methodologies</li>
            <li>SQL and Data Analysis</li>
            <li>Project Management</li>
            <li>Process Improvement Techniques</li>
        </ul>
        
        <h4>Recommended Certifications:</h4>
        <ul>
            <li>IIBA Certified Business Analysis Professional (CBAP)</li>
            <li>PMI Professional in Business Analysis (PMI-PBA)</li>
            <li>Certified Analytics Professional (CAP)</li>
            <li>IREB Certified Professional for Requirements Engineering</li>
            <li>Agile Analysis Certification (IIBA-AAC)</li>
        </ul>
    </div>
</div>
</div>
''', unsafe_allow_html=True)

# Tools section
st.markdown('''
<div class="dashboard-card" style="padding: 25px; margin-top: 30px; margin-bottom: 30px;">
<h2 style="color:#3366FF; border-bottom: 2px solid #3366FF; padding-bottom: 10px;">Essential Tools for Analysts</h2>

<div style="display: flex; margin-top: 20px;">
    <div style="flex: 1; padding-right: 15px;">
        <h3 style="color:#1E2A78;">Data Analysis Tools</h3>
        <ul>
            <li><strong>Programming Languages:</strong> Python, R, SQL</li>
            <li><strong>Data Visualization:</strong> Tableau, Power BI, QlikView</li>
            <li><strong>Statistical Analysis:</strong> SPSS, SAS, MATLAB</li>
            <li><strong>Database Management:</strong> MySQL, PostgreSQL, Oracle, MongoDB</li>
            <li><strong>Big Data Tools:</strong> Apache Hadoop, Spark, Hive</li>
            <li><strong>Data Integration:</strong> Alteryx, Talend, SSIS</li>
            <li><strong>Cloud Platforms:</strong> AWS, Google Cloud, Azure</li>
        </ul>
    </div>
    
    <div style="flex: 1; padding-left: 15px;">
        <h3 style="color:#1E2A78;">Business Analysis Tools</h3>
        <ul>
            <li><strong>Project Management:</strong> JIRA, Trello, Asana, MS Project</li>
            <li><strong>Process Modeling:</strong> Lucidchart, Visio, BPMN tools</li>
            <li><strong>Requirements Management:</strong> Confluence, ReqView, Modern Requirements</li>
            <li><strong>Wireframing & Prototyping:</strong> Balsamiq, Figma, InVision</li>
            <li><strong>Collaboration Tools:</strong> Slack, Microsoft Teams, Zoom</li>
            <li><strong>Documentation Tools:</strong> Confluence, SharePoint, Notion</li>
            <li><strong>Business Intelligence:</strong> Looker, Domo, SAP BusinessObjects</li>
        </ul>
    </div>
</div>
</div>

<!-- Footer -->
<div class="footer">
<p>© 2025 Business Analytics Dashboard | This guide is designed to help users understand the roles and workflows of Data Analysts and Business Analysts.</p>
</div>
''', unsafe_allow_html=True)