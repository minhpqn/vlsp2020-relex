import sys
import unittest

sys.path.append('.')

from prepare_data4phobert import Sentence, convert_to_phobert
from pyvi import ViTokenizer
from relex.datautils import load_id2label


class TestPhoBERTConversion(unittest.TestCase):

    def test_from_pyvi_sentence(self):
        # 22	24	26	27
        text = ('Bị xe tải cuốn vào gầm , cháu bé thoát chết thần kỳ Khoảng 17h chiều ngày 21/9 , '
                'tại ngã tư Phan Đình Phùng và Lê Lợi , thuộc phường Nghĩa Chánh , TP Quảng Ngãi xảy '
                'ra vụ tai nạn giao thông giữa xe ô tô tải và xe đạp .')
        tokenized_text = ViTokenizer.tokenize(text, sylabelize=False)
        # print(tokenized_text)
        sen = Sentence.from_pyvi_sentence(tokenized_text)
        # print(" ".join([s.text for s in sen.syllables]))
        # print(" ".join([tk.form for tk in sen.tokens]))
        self.assertEqual("Phan", sen.syllables[22].text)

        text = 'Một sĩ quan cảnh sát Hoàng gia Thái Lan đã khai với các nhà điều tra rằng ông đã “ được lệnh ” giúp cựu Thủ tướng Yingluck Shinawatra trốn thoát khỏi Thái Lan , theo Phó Thủ tướng Thái Lan Prawit Wongsuwon .'
        text = 'Nguồn : Báo Đất Việt Gia Bảo | 06:00 22/09/2017'
        tokenized_text = ViTokenizer.tokenize(text, sylabelize=False)
        print("Tokenized text:", tokenized_text)
        sen = Sentence.from_pyvi_sentence(tokenized_text)
        print(" ".join([s.text for s in sen.syllables]))
        print(" ".join([tk.form for tk in sen.tokens]))
    
    def test_convert_to_pho_bert(self):
        text = '2	22	24	26	27	LOCATION	LOCATION	Bị xe tải cuốn vào gầm , cháu bé thoát chết thần kỳ Khoảng 17h chiều ngày 21-9 , tại ngã tư Phan Đình Phùng và Lê Lợi , thuộc phường Nghĩa Chánh , TP Quảng Ngãi xảy ra vụ tai nạn giao thông giữa xe ô tô tải và xe đạp .'
        text = '2	5	6	2	4	PERSON	ORGANIZATION	Nguồn : Báo Đất Việt Gia Bảo | 06:00 22/09/2017'
        text = '0	41	42	39	40	PERSON	LOCATION	Một sĩ quan cảnh sát Hoàng gia Thái Lan đã khai với các nhà điều tra rằng ông đã “ được lệnh ” giúp cựu Thủ tướng Yingluck Shinawatra trốn thoát khỏi Thái Lan , theo Phó Thủ tướng Thái Lan Prawit Wongsuwon .'
        id2label = load_id2label('data/original_train_dev/VLSP2020_RE_SemEvalFormat/id2label.txt')
        new_text, text_with_markers = convert_to_phobert(text, id2label)
        print(text)
        print(new_text)
        print(text_with_markers)


if __name__ == '__main__':
    unittest.main()
