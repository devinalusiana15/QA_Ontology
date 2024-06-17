import nltk
import spacy
import json
import subprocess
import os.path
from django.db import models
from django.core.validators import MinLengthValidator
from tqdm import tqdm
from spacy.tokens import DocBin, Doc
from django.db import models, transaction
from rdflib import Graph, Namespace, URIRef
from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import LoginForm
from nltk.tag import pos_tag
from .forms import LoginForm, UploadFileForm
import requests
import fitz
from nltk.tag import pos_tag

import time
from django.utils.safestring import mark_safe
from django.db.models import Prefetch
from django.conf import settings
from django.contrib import messages
from django.db import transaction
from django.dispatch import receiver
from django.shortcuts import render, redirect, get_object_or_404
from rdflib import Graph, URIRef, Namespace, Literal
from owlready2 import onto_path, get_ontology, sync_reasoner

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')

class Uploader(models.Model):
    uploader_id = models.AutoField(primary_key=True)
    username = models.CharField(max_length=150, unique=True)
    password = models.CharField(max_length=128, validators=[MinLengthValidator(8)])

    def _str_(self):
        return self.username
    
    @staticmethod
    def login(request):
        if request.method == 'POST':
            form = LoginForm(request.POST)
            if form.is_valid():
                username = form.cleaned_data['username']
                password = form.cleaned_data['password']
                try:
                    user = Uploader.objects.get(username=username)
                    if user.password == password:
                        request.session['uploader_id'] = user.uploader_id
                        return redirect('uploadKnowledge')
                    else:
                        form.add_error(None, 'Invalid username or password')
                except Uploader.DoesNotExist:
                    form.add_error(None, 'Invalid username or password')
        else:
            form = LoginForm()
        return render(request, 'pages/uploaders/login.html', {'form': form})
    
    @staticmethod
    def logout(request):
        del request.session['uploader_id']
        return redirect('login')

class TextProcessing(models.Model):

    class Meta:
        abstract = True

    @staticmethod
    def remove_stopwords(doc):
        return ' '.join([token.text for token in doc if not token.is_stop])

    @staticmethod
    def pos_tagging_and_extract_verbs(text):
        doc = nlp_default(text)
        tokens = [token.text for token in doc]
        stop_words = nlp_default.Defaults.stop_words
        pos_tags = pos_tag(tokens)
        verbs = [word for word, pos in pos_tags if pos.startswith('VB') and word.lower() not in stop_words]
        return verbs

    @staticmethod
    def pos_tagging_and_extract_nouns(text):
        not_include = "coffee"
        doc = nlp_default(text)
        tokens = [token.text for token in doc]
        pos_tags = pos_tag(tokens)
        nouns = [word for word, pos in pos_tags if pos.startswith('NN') and word != not_include]
        return nouns

    @staticmethod
    def pos_tagging_and_extract_nouns_ontology(text):
        not_include = ["coffee", "definition"]
        doc = nlp_default(text)
        tokens = [token.text for token in doc]
        pos_tags = pos_tag(tokens)
        nouns = [word for word, pos in pos_tags if pos.startswith('NN')]

        if len(nouns) == 1 and nouns[0] == "coffee":
            return nouns
        else:
            nouns = [noun for noun in nouns if noun not in not_include]
            return nouns

    @staticmethod
    def find_answer_type(question):

        question = question.lower().split()

        question_keywords = ['what', 'when', 'where', 'who', 'why', 'how']

        if question[1] == "are" and question[0] in question_keywords:
            return ['axiom']
        elif question[0] in question_keywords:
            if 'where' in question:
                return ['LOC', 'GPE', 'CONTINENT', 'LOCATION']
            elif 'who' in question:
                return ['NORP', 'PERSON','NATIONALITY']
            elif 'when' in question:
                return ['DATE', 'TIME']
            elif 'what' in question:
                if 'definition' in question:
                    return ['definition']
                else:
                    return ['PERCENT', 'PRODUCT', 'VARIETY', 'METHODS', 'BEVERAGE', 'QUANTITY']
            elif 'how' in question:
                return ['direction']
        else:
            return "Pertanyaan tidak valid"

    @staticmethod
    def find_answer(answer_types, entities):
        answer_types_mapping = {
            'LOC': ['LOC','GPE', 'CONTINENT'],
            'PERSON': ['NORP', 'PERSON','NATIONALITY', 'JOB'],
            'DATE': ['DATE', 'TIME'],
            'PRODUCT': ['PRODUCT', 'VARIETY', 'METHODS', 'BEVERAGE', 'QUANTITY', 'DISTANCE', 'TEMPERATURE'],
        }
        for ent_text, ent_label in entities:
            for answer_type, labels in answer_types_mapping.items():
                if answer_type in answer_types and ent_label in labels:
                    return ent_text
        return "Tidak ada informasi yang ditemukan."

    @staticmethod
    def lemmatization(text):
        doc = nlp_default(text)
        filtered_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return filtered_tokens


class InvertedIndex(models.Model):
    class Meta:
        abstract = True

    @staticmethod
    def create_and_save_inverted_index(document):
        text = Documents.extract_text_from_pdf(document.document_path)
        sentences = text.split('.')
        
        with transaction.atomic():
            for sentence_index, sentence in enumerate(sentences, start=1):
                doc_details = DocDetails.objects.create(document=document, docdetail=sentence, position=sentence_index)

                sentence_doc = nlp_default(sentence)
                tokens = [token.text.lower() for token in sentence_doc if not token.is_stop and not token.is_punct]
                lemmatized_tokens = {token.text.lower(): token.lemma_ for token in sentence_doc if not token.is_stop and not token.is_punct}

                for token in tokens:
                    lemma = lemmatized_tokens.get(token)
                    term, created = Terms.objects.get_or_create(term=token)
                    if created or term.lemma is None:
                        term.lemma = lemma
                        term.save()
                    
                    PostingLists.objects.create(term=term, docdetail=doc_details)

    @staticmethod
    def retrieve_documents(keywords=None, nouns=None):
        relevant_documents = []
        relevant_sentences = []
        
        if keywords is None and nouns is None:
            return relevant_documents, relevant_sentences

        terms = Terms.objects.none()
        if keywords is not None:
            terms = Terms.objects.filter(term__in=keywords) | Terms.objects.filter(lemma__in=keywords)
        if nouns is not None:
            terms = terms | Terms.objects.filter(term__in=nouns) | Terms.objects.filter(lemma__in=nouns)

        if terms.exists():
            posting_entries = PostingLists.objects.filter(term__in=terms)
            for entry in posting_entries:
                doc_detail = entry.docdetail
                document_content = DocDetails.objects.filter(docdetail_id=doc_detail.docdetail_id).values_list('docdetail', flat=True).first()
                relevant_sentence = document_content
                
                relevant_documents.append({
                    'detail': entry.docdetail.docdetail_id,
                    'document_name': entry.docdetail.document_id,
                    'context': document_content,
                    'relevant_sentence': relevant_sentence,
                    'url': f'/document/{doc_detail.document_id}'
                })
                relevant_sentences.append(relevant_sentence)
        
        return relevant_documents, relevant_sentences
    
    @staticmethod
    def get_answer(question):
        keywords_verbs = TextProcessing.pos_tagging_and_extract_verbs(question)
        keywords_nouns = TextProcessing.pos_tagging_and_extract_nouns(question)
        response_text = f"Pertanyaan asli: {question}<br>Keywords (Verbs): {keywords_verbs}<br>Keywords (Nouns): {keywords_nouns}<br>"
        
        answer = "Tidak ada informasi yang ditemukan."
        
        search_result, relevant_sentences = Terms.retrieve_documents(keywords=keywords_verbs)
        
        if not search_result:
            search_result_nouns, relevant_sentences_nouns = Terms.retrieve_documents(nouns=keywords_nouns)
            search_result.extend(search_result_nouns)
            relevant_sentences.extend(relevant_sentences_nouns)
        
        if not search_result:
            lemmatized_verbs = TextProcessing.lemmatization(' '.join(keywords_verbs))
            lemmatized_nouns = TextProcessing.lemmatization(' '.join(keywords_nouns))

            search_result_lemmas, relevant_sentences_lemmas = Terms.retrieve_documents(keywords=lemmatized_verbs)
            
            if not search_result_lemmas:
                search_result_lemmas_nouns, relevant_sentences_lemmas_nouns = Terms.retrieve_documents(nouns=lemmatized_nouns)
                search_result_lemmas.extend(search_result_lemmas_nouns)
                relevant_sentences_lemmas.extend(relevant_sentences_lemmas_nouns)

            search_result.extend(search_result_lemmas)
            relevant_sentences.extend(relevant_sentences_lemmas)

        if search_result:
            for i, result in enumerate(search_result):
                doc_content = result['relevant_sentence']
                doc_entities = merge_entities(nlp_default(doc_content)).ents
                print(f"Entities in document {result['document_name']}: {doc_entities}")

                answer_types = TextProcessing.find_answer_type(question)
                print(f"Answer types: {answer_types}")

                answer = TextProcessing.find_answer(answer_types, [(ent.text, ent.label_) for ent in doc_entities])
                print(f"Answer found: {answer}")

                if answer != "Tidak ada informasi yang ditemukan.":
                    response_text += f"<br>Jawaban: {answer}"
                    break
                else:
                    response_text += f"<br>Jawaban tidak ditemukan dalam dokumen: {result['document_name']}"
                    refine = Refinements(question=question, answer=answer)
                    refine.save()
                    answer = "Tidak ada informasi yang ditemukan."
        else:
            response_text += "<br>Dokumen yang relevan tidak ditemukan."
            refine = Refinements(question=question, answer=answer)
            refine.save()
        
        
        predicate = keywords_verbs
        rdf_output = Ontology.get_rdf_answer(predicate)

        context = {'response_text': response_text, 'related_articles': relevant_sentences}
        print(context)
        extra_info = Ontology.get_extra_information(answer.replace(" ", "_"))
        return answer, search_result, extra_info, rdf_output

class Documents(models.Model):
    document_id = models.AutoField(primary_key=True)  # ID unik untuk setiap dokumen
    document_name = models.CharField(max_length=255)  # Nama atau judul dokumen
    document_path = models.CharField(max_length=255)  # Isi dari dokumen tersebut

    def _str_(self):
        return self.document_name
    
    @staticmethod
    def add_knowledge(request):
        return render(request, 'pages/seekers/addKnowledge.html')

    @staticmethod
    def upload_knowledge(request):
        if request.method == 'POST':
            form = UploadFileForm(request.POST, request.FILES)
            if form.is_valid():
                uploaded_file = request.FILES['file']
                if uploaded_file.content_type != 'application/pdf':
                    messages.error(request, 'File must be in PDF format.')
                else:
                    upload_dir = os.path.join(settings.BASE_DIR, 'kms_app/uploaded_files')
                    if os.path.exists(os.path.join(upload_dir, uploaded_file.name)):
                        messages.error(request, 'File already exists.')
                    else:
                        new_document = Documents(document_name=uploaded_file.name, document_path='kms_app/uploaded_files/'+uploaded_file.name)
                        new_document.save()

                        Documents.handle_uploaded_file(uploaded_file)
                        InvertedIndex.create_and_save_inverted_index(new_document)

                        text = Documents.extract_text_from_pdf(new_document.document_path)
                        document = [merge_entities(nlp_custom(sentence)) for sentence in text.split('.') if sentence.strip()]

                        ontology = Ontology.generate_ontology(document)
                        Ontology.save_ontology(ontology)

                        messages.success(request, 'New knowledge is added successfully')
                        return render(request, 'pages/uploaders/uploadersAddKnowledge.html')
            else:
                messages.error(request, 'Failed to add new knowledge')
        else:
            form = UploadFileForm()
        return render(request, 'pages/uploaders/uploadersAddKnowledge.html', {'form': form})
    
    @staticmethod
    def handle_uploaded_file(file):
        upload_dir = os.path.join(settings.BASE_DIR, 'kms_app/uploaded_files')
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        with open(os.path.join(upload_dir, file.name), 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

    @staticmethod
    def extract_text_from_pdf(context_path):
        text = ""
        try:
            with fitz.open(context_path) as doc:
                for page in doc:
                    text += page.get_text()
        except Exception as e:
            print("Error:", e)
        return text

class Terms(InvertedIndex):
    term = models.CharField(max_length=255, unique=True, primary_key=True)  # Term atau kata kunci yang muncul dalam dokumen
    lemma = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return self.term

    class Meta:
        indexes = [
            models.Index(fields=['term']),  # Menambahkan indeks pada kolom term
        ]
    
class DocDetails(InvertedIndex):
    docdetail_id = models.AutoField(primary_key=True)  # ID unik untuk setiap dokumen
    document = models.ForeignKey(Documents, on_delete=models.CASCADE)  # ID dokumen yang merujuk ke Tabel Dokumen
    docdetail = models.CharField(max_length=255)  # Isi dari dokumen tersebut
    position = models.IntegerField()

    def _str_(self):
        return self.docdetail

class PostingLists(InvertedIndex):
    postlist_id = models.AutoField(primary_key=True)  # ID unik untuk setiap entri dalam posting list
    term = models.ForeignKey(Terms, on_delete=models.CASCADE,to_field='term', db_column='term')  # ID term yang merujuk ke Tabel Term
    docdetail = models.ForeignKey(DocDetails, on_delete=models.CASCADE)  # Frekuensi kemunculan term dalam dokumen tertentu

    def _str_(self):
        return f"{self.term} - {self.docdetail}"
    
    class Meta:
        indexes = [
            models.Index(fields=['term']),  # Menambahkan indeks pada kolom term
            models.Index(fields=['docdetail']),
        ]

class Refinements(models.Model):
    refinement_id = models.AutoField(primary_key=True)
    question = models.CharField(max_length=255)
    answer = models.CharField(max_length=255)

class Ontology(models.Model):
    class Meta:
        abstract = True

    @staticmethod   
    def get_fuseki_data(query_string):
        endpoint = "http://localhost:3030/Kopi/query"

        # send SPARQL query
        r = requests.get(endpoint, params={'query': query_string})
        
        # get query results
        results = []
        if r.status_code == 200:
            response = r.json()
            for result in response['results']['bindings']:
                formatted_result = {}
                for key in result.keys():
                    formatted_result[key] = result[key]['value']
                results.append(formatted_result)

        return results

    @staticmethod
    def generate_ontology(doc_ontology):
        cleaned_sentences = []
        for sent in doc_ontology:
            cleaned_sentences.append(TextProcessing.remove_stopwords(sent))

        clean_ents=[]
        for sent in cleaned_sentences:
            clean_ents.append(merge_entities(nlp_custom(sent)))

        # Proses pembuatan ontologi
        ontology = ""

        classes = set()
        object_properties = set()

        for sent in clean_ents:
            prev_entity = None
            for ent in sent.ents:
                if ent.label_ != '':
                    if ent.label_ == 'VERB':
                        if prev_entity:
                            # Next entity
                            next_entity = None
                            for next_ent in sent.ents:
                                if next_ent.start > ent.start:
                                    next_entity = next_ent
                                    break
                            if next_entity:
                                if next_entity.label_ in ["DATE", "TIME"]:
                                    obj_prop = f"{ent.text}_on"
                                elif next_entity.label_ in ["LOC", "GPE"]:
                                    obj_prop = f"{ent.text}_in"
                                elif next_entity.label_ in ["NORP", "PERSON"]:
                                    obj_prop = f"{ent.text}_by"
                                elif prev_entity.label_ == next_entity.label_:
                                    obj_prop = f"{ent.text}"                             
                                else:
                                    obj_prop = f"{ent.text}"
                            
                                object_properties.add(obj_prop)

                                # Penentuan Domain
                                ontology += f"""
                                <http://www.semanticweb.org/ariana/coffee#{obj_prop}> rdfs:domain <http://www.semanticweb.org/ariana/coffee#{prev_entity.label_}> .
                                """
                                # Penentuan Range
                                ontology += f"""
                                    <http://www.semanticweb.org/ariana/coffee#{obj_prop}> rdfs:range <http://www.semanticweb.org/ariana/coffee#{next_entity.label_}> .
                                """
                                # Individual - Object Property - Individual
                                ontology += f"""
                                    <http://www.semanticweb.org/ariana/coffee#{prev_entity.text.replace(" ", "_")}> coffee:{obj_prop} <http://www.semanticweb.org/ariana/coffee#{next_entity.text.replace(" ", "_")}> .
                                """
                        prev_entity = None  # Reset prev_entity
                    else:
                        prev_entity = ent

        for sent in clean_ents:
            for ent in sent.ents:
                if ent.label_ != '':
                    individual_name = ent.text.replace(" ", "_")
                    if ent.label_ != 'VERB':
                        classes.add(ent.label_)
                        ontology += f"""
                        <http://www.semanticweb.org/ariana/coffee#{individual_name}> rdf:type <http://www.semanticweb.org/ariana/coffee#{ent.label_}> .
                        """
        return ontology

    @staticmethod
    def save_ontology(ontology):
        owl_directory = os.path.join(settings.BASE_DIR, 'kms_app/owl_file')
        file_path = os.path.join(owl_directory, "Kopi.owl")
        with open(file_path, "a") as output_file:
            output_file.write(ontology)

    @staticmethod
    def get_extra_information(answer):
        COFFEE = Namespace("http://www.semanticweb.org/ariana/coffee#")
        g = Graph()
        g.bind("coffee", COFFEE)

        query = f"""
        PREFIX coffee: <http://www.semanticweb.org/ariana/coffee#>
        SELECT ?p ?o ?s WHERE {{
        {{ coffee:{answer} ?p ?o.
            FILTER (!CONTAINS(LCASE(STR(?p)), "type"))
        }}
        UNION
        {{ ?s ?p coffee:{answer}.
            FILTER (!CONTAINS(LCASE(STR(?p)), "type"))
        }}
        }}
        """
        results = Ontology.get_fuseki_data(query)

        text_response = ""
        if results:
            for row in results:
                predicate_name = row.get('p', '').split('#')[-1].replace("_", " ") if row.get('p') else None
                object_name = row.get('o', '').split('#')[-1].replace("_", " ") if row.get('o') else None
                subject_name = row.get('s', '').split('#')[-1].replace("_", " ") if row.get('s') else None

                if predicate_name and object_name:
                    g.add((COFFEE[answer], URIRef(row['p']), URIRef(row['o'])))
                    text_response += f"{answer.replace('_', ' ')} {predicate_name} {object_name}. "
                if predicate_name and subject_name:
                    g.add((URIRef(row['s']), URIRef(row['p']), COFFEE[answer]))
                    text_response += f"{subject_name} {predicate_name} {answer.replace('_', ' ')}. "

            rdf_output = g.serialize(format='turtle')
        else:
            rdf_output = None
        
        extra_info = {
            'answer': answer,
            'text_response': text_response,
            'rdf_output': rdf_output, 
        }

        return extra_info

    @staticmethod
    def get_annotation(question,annotation):

        keywords_nouns = TextProcessing.pos_tagging_and_extract_nouns_ontology(question)

        noun = "_".join(keywords_nouns)
        print(noun)

        response = ""

        query = f"""
        PREFIX coffee: <http://www.semanticweb.org/ariana/coffee#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?s WHERE {{
        coffee:{noun} rdfs:{annotation[0]} ?s
        }}
        """

        try:
            results = Ontology.get_fuseki_data(query)
        except Exception as e:
            print(f"Error executing query: {e}")
            return "Error executing query"

        if results:
            for row in results:
                response = row['s'].replace("\n", "<br>")
        else:
            response = "Tidak ada jawaban"

        return response

    @staticmethod
    def get_instances(noun):
        onto_path.append(os.path.join(settings.BASE_DIR, 'kms_app/owl_file'))
        onto = get_ontology("Kopi.rdf").load()

        # Mengaktifkan reasoner
        sync_reasoner()

        # Noun sebagai class yang dicari
        keyword_noun = "".join(noun)

        # Mencari kelas berdasarkan kata kunci
        cls = onto[keyword_noun]
        if not cls:
            return "Class not found"

        instances = list(cls.instances())

        response = f"<br>These are the {keyword_noun}:"

        if instances:
            # Mendapatkan properti yang sama pada setiap instance
            common_properties = None
            for instance in instances:
                instance_properties = set()
                for prop in instance.get_properties():
                    instance_properties.add(prop.name)

                if common_properties is None:
                    common_properties = instance_properties
                else:
                    common_properties = common_properties.intersection(instance_properties)

            # Menampilkan instance - common_properties - value
            if common_properties:
                for prop_name in common_properties:
                    for instance in instances:
                        for prop in instance.get_properties():
                            if prop.name == prop_name:
                                for value in prop[instance]:
                                    response += f"<br>- {instance.name.replace('_', ' ')} {prop.name} {value.name.replace('_', ' ')} "
        else:
            response += "No instances found."

        return response
    
    @staticmethod
    def get_rdf_answer(predicate):
        COFFEE = Namespace("http://www.semanticweb.org/ariana/coffee#")
        g = Graph()
        g.bind("coffee", COFFEE)
        
        if isinstance(predicate, list) and len(predicate) == 1:
            predicate = predicate[0]

        query = f"""
        PREFIX coffee: <http://www.semanticweb.org/ariana/coffee#>
        SELECT ?s ?o WHERE {{
        ?s coffee:{predicate} ?o .
        }}
        """
        results = Ontology.get_fuseki_data(query)

        if results:
            for row in results:
                object_name = row.get('o', '').split('#')[-1].replace("_", " ") if row.get('o') else None
                subject_name = row.get('s', '').split('#')[-1].replace("_", " ") if row.get('s') else None
                
                if subject_name and object_name:
                    g.add((URIRef(row['s']), COFFEE[predicate], URIRef(row['o'])))
        else:
            return None

        rdf_output = g.serialize(format='turtle')

        return rdf_output

# Model NER Default
nlp_default = spacy.load("en_core_web_sm")

# Model NER Custom
model_path = "kms_app/training/model-best"
if os.path.exists(model_path):
    nlp_custom = spacy.load(model_path)
else:
    # Jika model khusus tidak ditemukan, buat model kosong
    nlp_custom = spacy.blank("en")  

    train_data_path = 'kms_app/training/train_data.json'

    # Cek apakah file JSON ada
    if os.path.exists(train_data_path):
        # Buka file JSON yang berisi data train
        with open(train_data_path, 'r', encoding='utf-8') as f:
            TRAIN_DATA = json.load(f)

        # Filter anotasi untuk menghapus entri null
        filtered_annotations = [annotation for annotation in TRAIN_DATA['annotations'] if annotation is not None]

        # Inisialisasi objek DocBin untuk menyimpan dokumen Spacy
        db = DocBin()

        # Iterasi melalui anotasi yang difilter
        for text, annot in tqdm(filtered_annotations):
            # Buat objek dokumen Spacy dari teks
            doc = nlp_default.make_doc(text)
            ents = []

            # Iterasi melalui entitas yang diberikan dalam anotasi
            for start, end, label in annot["entities"]:
                # Buat objek span untuk entitas
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
                if span is None:
                    print("Melewati entitas")
                else:
                    ents.append(span)

            # Atur entitas yang ditemukan dalam dokumen
            doc.ents = ents
            # Tambahkan dokumen ke DocBin
            db.add(doc)

        # Simpan data pelatihan ke disk dalam format Spacy
        db.to_disk("kms_app/training/training_data.spacy")

        # Eksekusi perintah setelah penyimpanan data ke disk
        init_config_args = "init config kms_app/training/config.cfg --lang en --pipeline ner --optimize efficiency"
        train_args = "train kms_app/training/config.cfg --output kms_app/training/ --paths.train kms_app/training/training_data.spacy --paths.dev kms_app/training/training_data.spacy"

        # Jalankan perintah untuk inisialisasi konfigurasi
        subprocess.run(["python", "-m", "spacy"] + init_config_args.split())

        # Jalankan perintah untuk melatih model
        subprocess.run(["python", "-m", "spacy"] + train_args.split())

def merge_entities(doc):
    combined_entities = []
    entities_custom = {}
    entities_default = {}

    # Entitas dari model NER default
    for ent in nlp_default(doc.text).ents:
        entities_default[(ent.start_char, ent.end_char)] = (ent.label_, ent.text)

    # Entitas dari model NER custom
    for ent in nlp_custom(doc.text).ents:
        entities_custom[(ent.start_char, ent.end_char)] = (ent.label_, ent.text)

    # Gabungkan entitas
    combined_entities = entities_custom.copy()

    for (start_char, end_char), (label, text) in entities_default.items():
        overlap = False
        for (start_custom, end_custom) in entities_custom.keys():
            if start_char < end_custom and start_custom < end_char:
                overlap = True
                break
        if not overlap:
            combined_entities[(start_char, end_char)] = (label, text)

    print(combined_entities)
    # Create a list to store spans
    spans = []

    # Create spans using character offsets directly
    for (start_char, end_char), (label, text) in combined_entities.items():
        span = doc.char_span(start_char, end_char, label=label)
        if span is None:
            print(f"Skipping entity: {text}")
        else:
            spans.append(span)

    # Create a new document with the combined entities
    merged_doc = Doc(doc.vocab, words=[token.text for token in doc])
    merged_doc.ents = spans

    return merged_doc

