from django.urls import path
from kms_app import views
from django.shortcuts import redirect

urlpatterns = [
    # URL Default ke Home
    path('', lambda request: redirect('QAontology/', permanent=False)),
    
    path('QAontology/', views.home_ontology, name='QAontology'), 
    path('QAinverted/', views.home_inverted, name='QAinverted'), 
    path('articles/', views.articles_view, name='articles'), 
    path('articles/<int:document_id>/', views.detail_article_view, name='detailArticle'), 
    path('document/<int:document_id>/', views.detail_article_view, name='detailArticle'), 
    path('uploadKnowledge/', views.add_knowledge_view, name='addKnowledge'),
    path('uploaders/uploadKnowledge/', views.upload_knowledge_view, name='uploadKnowledge'),
    path('login/', views.login_view, name="login"),
    path('logout/', views.logout_view, name="logout"),
]
