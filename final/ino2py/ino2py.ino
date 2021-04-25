int buttonPin = 6;
int ledPin;
int i;
int count = 0;
String str;
#include <Servo.h>
Servo myservo;
void setup() {
  myservo.attach(9);
  Serial.begin(9600);
  myservo.write(90);
  pinMode(ledPin, OUTPUT);
  pinMode(buttonPin, INPUT);
}
void loop() {
  if (Serial.available()) {
    // 讀取傳入的字串直到"\n"結尾
    str = Serial.readStringUntil('\n');
    if (str == "on" && count == 0) {     
      myservo.write(0); // 使用write，傳入角度，從0度轉到180度
      count = 1;
    }

    if (digitalRead(buttonPin)) {
      count = 0;
      Serial.println(count);
      myservo.write(90);
    }
  }
}
