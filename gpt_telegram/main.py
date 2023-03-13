from dotenv import dotenv_values
import openai
import sqlite3
import telebot
from requests.exceptions import ReadTimeout
from openai.error import InvalidRequestError

env = {**dotenv_values("./key.env")}  # создать файл ****.env где будет токен бота и api-ключ openai

keys_chat_gpt = env["key1"]  # ключ  openai
bot = telebot.TeleBot(env["token"])  # токен бота в телеграм
db_link = "gpt.db"  # простенькая база данных для сохранения логов.


# функция для записи в бд
def write_into_db(text):
    conn = sqlite3.connect(db_link)
    cursor = conn.cursor()
    id = cursor.execute("SELECT id FROM users WHERE chat_id = ?", (str(text.chat.id),)).fetchone()
    if id:
        try:
            cursor.execute("UPDATE users SET last_msg=? ,last_log=? ,WHERE chat_id = ?",
                           (text.text, text.date, text.chat.id))
        except:
            conn.commit()
            conn.close()
            bot.send_message(env["token"], f"Ошибка при добавлении ваших данных в базу данных: {text.chat.id}")
    else:
        try:
            cursor.execute("INSERT INTO users (chat_id,last_login,username,last_message) VALUES (?,?,?,?)",
                           (str(text.id), str(text.date), (text.chat.username if text.chat.username else "-"),
                            text.text))
        except:
            conn.commit()
            conn.close()
            bot.send_message(env["token"], f"Ошибка при добавлении ваших данных в базу данных: {text.chat.id}")
        conn.commit()
        conn.close()


# проверка ответа бота на длину, если не вмещается выводить ответ поочередно.
def check_length(answer, list_of_answers):
    if 4090 < len(answer) < 409000:
        list_of_answers.append(answer[0:4090] + "...")
        check_length(answer[4091:], list_of_answers)
    else:
        list_of_answers.append(answer[0:])
        return list_of_answers


# собственно запрос к chat-gpt
def chat_for_gpt(message):
    engine = "text-davinci-003"
    try:
        completion = openai.Completion.create(engine=engine, prompt=message.text, temperature=0.5, max_tokens=1000)
        print(completion.choices[0]["text"])
        answers = check_length(completion.choices[0]["text"], [])
        if answers:
            for i in answers:
                bot.send_message(message.chat.id, i)
        else:
            chat_for_gpt(message)
    except ReadTimeout:
        bot.send_message(message.chat.id, "Бот перегружен запросами,пожалуйста подождите")
    except InvalidRequestError:
        bot.send_message(message.chat.id, "Максимум 1000 символов в запросе")


def create_db():
    conn = sqlite3.connect(db_link)
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS users(
                    id INTEGER PRImARY KEY AUTOINCREMENT,
                    chat_id TEXT,
                    last_login TEXT,
                    username TEXT,
                    last_message TEXT)
                   """)
    conn.commit()
    conn.close()


@bot.message_handler(commands=["start"])
def start(message):
    text = """ ЗДАРОВА ЗАЕБАЛ.
 лан,шучу ,привет славик,я тут крч решил позаниматься хуйней и написал сранного бота ,объединив его со своим GPT
 ну типо хайпую тоже)
 пиши текст,поидее должен работать как стандартный чат гпт ,но есть две особенности:
    1.я пока не разобрался как поддерживать связь в диалоге,так что он только отвечает на вопросы.
    2. ставь знак препинания в конце(.,?!) иначе он додумает предложение за тебя и ответит уже на него:D
    крч  потестируй его немного,если работать перестанет напиши мне.Это моей первый бот,так что багов должно быть 
    дохуища
    """
    write_into_db(message)
    bot.send_message(message.chat.id, text)


@bot.message_handler(content_types=["text"])
def message_from_user(message):
    openai.api_key = keys_chat_gpt
    write_into_db(message)
    chat_for_gpt(message)


if __name__ == "__main__":
    create_db()
    target = bot.infinity_polling()
