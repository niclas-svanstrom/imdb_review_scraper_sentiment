import sys
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QLineEdit, QTextBrowser, QRadioButton, QPlainTextEdit, QScrollArea, QProgressBar
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from bs4 import BeautifulSoup
import requests
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# Ladda den tränade modellen och tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('./model/Tokenizer')
bert_model = TFBertForSequenceClassification.from_pretrained('./model/Model')

class ModelInitializationThread(QThread):
    done_signal = pyqtSignal()

    def run(self):
        # Körs i en separat tråd vid programstart
        # Dummy-operation för att förbereda modellen
        dummy_input = ["This is a dummy input."]  # Använd en verklig dummy-input
        Input_ids, Token_type_ids, Attention_mask = bert_tokenizer.batch_encode_plus(dummy_input,
                                                                        padding=True,
                                                                        truncation=True,
                                                                        max_length=110,
                                                                        return_tensors='tf').values()

        _ = bert_model.predict([Input_ids, Token_type_ids, Attention_mask])
        self.done_signal.emit()

class ReviewScraperApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMDb Review Analyzer")
        self.init_ui()

        self.model_thread = ModelInitializationThread()
        self.model_thread.done_signal.connect(self.model_initialization_done)
        self.model_thread.start()

    def model_initialization_done(self):
        # Kallas när tråden har slutfört modellinitialiseringen
        self.analyze_button.setDisabled(False)

    def init_ui(self):
        # Skapa etikett för att visa resultatet
        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.result_label.setStyleSheet("color: red;")

        # Skapa radio-knappar för att välja typ av input
        self.link_radio = QRadioButton("Länk", self)
        self.review_radio = QRadioButton("Recension", self)

        # Skapa en knapp för att starta analysen
        self.analyze_button = QPushButton("Analysera", self)
        self.analyze_button.clicked.connect(self.analyze_input)
        self.analyze_button.setDisabled(True)  # Inaktivera vid start

        # Skapa en QPlainTextEdit för recensionen
        self.review_input = QPlainTextEdit(self)
        self.review_input.setPlaceholderText("Skriv din recension här")
        # self.review_input.setFixedHeight(150)  # Ange önskad höjd
        self.review_input.setDisabled(True)  # Inaktivera vid start

        # Skapa en QLineEdit för länken
        self.link_input = QLineEdit(self)
        self.link_input.setPlaceholderText("Ange IMDb-filmlänk här")
        self.link_input.setDisabled(True)  # Aktivera vid start

        # Skapa en QScrollArea för att möjliggöra rullning av recensionen
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.review_input)

        self.result_text = QTextBrowser(self)
        self.result_text.setAlignment(Qt.AlignBottom)
        # self.result_text.hide()  # Göm result_text vid start
        self.result_text.setOpenExternalLinks(True)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.hide()

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.result_label)
        layout.addWidget(self.link_radio)
        layout.addWidget(self.link_input)
        layout.addWidget(self.review_radio)
        layout.addWidget(self.scroll_area)
        layout.addWidget(self.analyze_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.result_text)

        self.setLayout(layout)

        self.link_input.textChanged.connect(self.check_input)
        self.review_input.textChanged.connect(self.check_input)
        self.link_radio.toggled.connect(self.check_input)
        self.review_radio.toggled.connect(self.check_input)

        self.resize(400, 500)

    def clear_results(self):
        # Rensa resultaten och göm result_text
        self.result_text.clear()
        # self.result_text.hide()

    def toggle_input(self, enabled_widget, disabled_widget, radio_button):
        # Inaktivera vald widget och aktivera den andra
        enabled_widget.setDisabled(False)
        disabled_widget.setDisabled(True)
        radio_button.setDisabled(False)

    def analyze_input(self):
        # Hämta input från länk- eller recensionsrutan beroende på val
        if self.link_radio.isChecked():
            input_text = self.link_input.text()
        elif self.review_radio.isChecked():
            input_text = self.review_input.toPlainText()

        if not input_text:
            self.result_text.clear()
            self.result_label.setText("Ange en giltig IMDb-filmlänk eller recension.")
            return

        # Kolla vilken typ av input som valts
        if self.link_radio.isChecked():
            # Om det är en länk, skrapa recensionerna och analysera
            imdb_id = self.extract_imdb_id(input_text)
            if not imdb_id:
                self.result_text.clear()
                self.result_label.setText("Kunde inte extrahera IMDb-ID från länken.")
                return

            reviews_url = f"https://www.imdb.com/title/{imdb_id}/reviews?sort=submissionDate&dir=desc&ratingFilter=0"
            reviews = self.scrape_reviews_from_url(reviews_url)
            self.analyze_reviews(reviews)
        elif self.review_radio.isChecked():
            # Om det är en recension, analysera direkt
            sentiment = self.analyze_review(input_text)
            self.display_result(sentiment)

    def extract_imdb_id(self, film_url):
        # Extrahera IMDb-ID från film-URL:en
        try:
            imdb_id = film_url.split("/")[4]
            return imdb_id
        except IndexError:
            return None

    def scrape_reviews_from_url(self, reviews_url):
        reviews = []
        try:
            # Skicka en GET-förfrågan till recensions-URL:en och hämta sidans HTML
            response = requests.get(reviews_url)
            response.raise_for_status()  # Kasta ett undantag om något går fel med förfrågan

            # Skrapa recensioner från sidan
            soup = BeautifulSoup(response.text, 'html.parser')
            review_containers = soup.find_all("div", class_="review-container")

            for review_container in review_containers:
                review_text_element = review_container.find("div", class_="text show-more__control")
                review_title_element = review_container.find("a", class_="title")

                if review_text_element and review_title_element:
                    review_text = review_text_element.get_text(strip=True)
                    review_url = review_title_element.get("href")
                    review_title = review_title_element.get_text(strip=True)
                    reviews.append({"text": review_text, "url": review_url, "title": review_title})

        except Exception as e:
            print(f"Error scraping reviews: {e}")

        return reviews
    
    def display_result(self, sentiment):
        self.clear_results()
        color = 'green' if sentiment == 'Positive' else 'red'
        formatted_sentiment = f'<font color="{color}">{sentiment}</font>'
        self.result_text.append(f"Recensionen är {formatted_sentiment}")

        self.result_text.verticalScrollBar().setValue(0)

    def analyze_reviews(self, reviews):
        # Analysera varje recension och visa resultatet i textrutan
        self.clear_results()

        self.progress_bar.show()

        self.completed = 0
        self.length = 100/len(reviews)

        for index, review in enumerate(reviews, start=1):
            sentiment = self.analyze_review(review['text'])
            color = 'green' if sentiment == 'Positive' else 'red'
            formatted_sentiment = f'<font color="{color}">{sentiment}</font>'
            imdb_link = f'https://www.imdb.com{review["url"]}'
            self.result_text.append(f"Recension: <a href='{imdb_link}'>{review['title']}</a> är {formatted_sentiment} <br> {'='*30} <br> Recensionstext: <br> {review['text']} <br><br>")
            self.completed += self.length
            self.progress_bar.setValue(int(self.completed))
        self.progress_bar.hide()
        self.result_text.verticalScrollBar().setValue(0)

    def analyze_review(self, Review):
        # Använd den tränade modellen för att analysera recensionen
        if not isinstance(Review, list):
            Review = [Review]
    
        Input_ids, Token_type_ids, Attention_mask = bert_tokenizer.batch_encode_plus(Review,
                                                                                padding=True,
                                                                                truncation=True,
                                                                                max_length=300,
                                                                                return_tensors='tf').values()

        prediction = bert_model.predict([Input_ids, Token_type_ids, Attention_mask])
        pred_labels = tf.argmax(prediction.logits, axis=1)
        label = {
            1: 'Positive',
            0: 'Negative'
        }
        # Convert the TensorFlow tensor to a NumPy array and then to a list to get the predicted sentiment labels
        pred_labels = [label[i] for i in pred_labels.numpy().tolist()]
        return pred_labels[0]

    def check_input(self):
        if self.link_radio.isChecked():
            self.link_input.setDisabled(False)
            self.review_input.setDisabled(True)
        elif self.review_radio.isChecked():
            self.link_input.setDisabled(True)
            self.review_input.setDisabled(False)
        # Aktivera "Analysera"-knappen endast om en av checkboxarna är ikryssad och motsvarande inmatningsruta inte är tom
        if (self.link_radio.isChecked() and self.link_input.text()) or \
           (self.review_radio.isChecked() and self.review_input.toPlainText()):
            self.analyze_button.setDisabled(False)
        else:
            self.analyze_button.setDisabled(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ReviewScraperApp()
    ex.show()
    sys.exit(app.exec_())
