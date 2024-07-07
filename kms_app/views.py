import os

import time
from django.utils.safestring import mark_safe
from django.shortcuts import render
from django.utils.safestring import mark_safe

from .models import (
    Uploader,
    Documents,
    TextProcessing,
    Ontology,
    InvertedIndex,
    merge_entities,
    nlp_custom
)

def login_view(request):
    return Uploader.login(request)

def logout_view(request):
    return Uploader.logout(request)

def articles_view(request):
    documents = Documents.objects.all()
    print(documents)
    
    context = []

    for document in documents:
        extracted_text = Documents.extract_text_from_pdf(document.document_path)

        truncated_text = extracted_text[:1000]

        article_data = {
            'doc_name': os.path.splitext(document.document_name)[0],
            'context': truncated_text.replace('_', '-') + '...',
            'full_path': document.document_path,
            'id': document.document_id
        }

        context.append(article_data)

    return render(request, 'pages/articles.html', {'articles': context})

def detail_article_view(request, document_id):
    
    document = Documents.objects.get(document_id=document_id)
    extracted_text = Documents.extract_text_from_pdf(document.document_path)

    article_data = {
        'doc_name': os.path.splitext(document.document_name)[0],
        'full_text': extracted_text.replace('_', '-')
    }

    return render(request, 'pages/detailArticle.html', {'article': article_data})

def add_knowledge_view(request):
    return Documents.add_knowledge(request)

def upload_knowledge_view(request):
    return Documents.upload_knowledge(request)

def home(request):
    if request.method == 'POST':
        start_time = time.time()
        search_query = request.POST.get('question')

        if not search_query.endswith('?'):
            context = {
                'question': search_query,
                'answer': 'Not the type of question. Please enter the question again.',
                'related_articles': None,
                'extra_info': None,
                'rdf_output': None
            }
            return render(request, 'Home.html', context)

        try:
            answer_types = TextProcessing.find_answer_type(search_query)
        except IndexError:
            context = {
                'question': search_query,
                'answer': 'Not the type of question. Please enter the question again.',
                'related_articles': None,
                'extra_info': None,
                'rdf_output': None
            }
            return render(request, 'Home.html', context)

        annotation_types = ['definition', 'direction']
        if answer_types == 'axiom':
            keyword_noun = TextProcessing.pos_tagging_and_extract_nouns(search_query)
            print(keyword_noun)
            answer, rdf_output = Ontology.get_instances(keyword_noun)
            context = {
                'question': search_query,
                'answer': mark_safe(answer),
                'rdf_output': rdf_output
            }
        elif answer_types == 'confirmation':
            confirmation, rdf_output = Ontology.confirmation(search_query)
            context = {
                'question': search_query,
                'answer': confirmation,
                'rdf_output': rdf_output
            }
        elif answer_types in annotation_types:
            answer, rdf_output = Ontology.get_annotation(search_query, answer_types)
            context = {
                'question': search_query,
                'answer': mark_safe(answer),
                'related_articles': None,
                'extra_info': None,
                'rdf_output': rdf_output
            }
        else:
            answer, rdf_output, extra_info = Ontology.get_answer_ontology(search_query, answer_types)            
            context = {
                'question': search_query,
                'answer': answer,
                'extra_info': extra_info,
                'rdf_output': rdf_output
            }
        
        end_time = time.time()
        response_time = end_time - start_time
        response_time = round(response_time, 2)
        
        context['response_time'] = response_time

        return render(request, 'Home.html', context)
    else:
        return render(request, 'Home.html', {'related_articles': []})