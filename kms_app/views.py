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

def home_ontology(request):
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
            keyword_nouns = TextProcessing.pos_tagging_and_extract_nouns(search_query)
            # Noun sebagai class yang dicari
            keyword_noun = "_".join(keyword_nouns)
            print(keyword_noun)
            answer = Ontology.get_instances(keyword_noun)
            context = {
                'question': search_query,
                'answer': mark_safe(answer)
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
            answer = Ontology.get_answer_ontology(search_query, answer_types)            
            context = {
                'question': search_query,
                'answer': answer
            }
        
        end_time = time.time()
        response_time = end_time - start_time
        response_time = round(response_time, 2)
        
        context['response_time'] = response_time

        return render(request, 'Home.html', context)
    else:
        return render(request, 'Home.html', {'related_articles': []})
    
def home_inverted(request):
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
            answer = InvertedIndex.axiom_inverted(search_query)
            context = {
                'question': search_query,
                'answer': mark_safe(answer)
            }
        elif answer_types in annotation_types:
            answer = InvertedIndex.get_definition_direction(search_query)
            context = {
                'question': search_query,
                'answer': mark_safe(answer),
            }
        else:
            answer = InvertedIndex.get_answer_inverted(search_query)            
            context = {
                'question': search_query,
                'answer': answer,
            }
        
        end_time = time.time()
        response_time = end_time - start_time
        response_time = round(response_time, 2)
        
        context['response_time'] = response_time

        return render(request, 'Home.html', context)
    else:
        return render(request, 'Home.html', {'related_articles': []})