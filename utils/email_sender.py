
import smtplib
from email.mime.text import MIMEText
from email.header import Header

class MsgSender():
    def __init__(self,mail_host,sender,recievers,passwd):
        self.mail_host = mail_host
        self.sender = sender
        self.recievers = recievers
        self.passwd = passwd

    def send_msg(self, msg):
        try:
            smtpObj = smtplib.SMTP_SSL(self.mail_host, 465)
            smtpObj.login(self.sender, self.passwd)
            msg = None
            for recivers in self.recievers:
                msg = MIMEText(msg, 'plain', 'utf-8')
                msg['From'] = Header(self.sender,'utf-8')
                msg['TO'] = Header(recivers,'utf-8')
                subject = 'Training Information'
                msg['Subject'] = Header(subject,'utf-8')
                smtpObj.sendmail(self.sender, self.recievers, msg.as_string())
                print(f'send email sucessed to {recivers}')
            smtpObj.quit()
        except smtplib.SMTPException as e:
            print(e)

if __name__ == '__main__':
    email_sender = MsgSender(
        mail_host='smtp.163.com',
        sender='wang_yunlong000@163.com',
        recievers=['354546602@qq.com','yunlong.wang.de@outlook.com'],
        passwd='GFNFBWQISTTGMSWK'
    )
    email_sender.send_msg('123')
