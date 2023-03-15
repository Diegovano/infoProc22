#include <ESP32SPISlave.h>
#include <WiFi.h>
#include <time.h>
#include <WString.h>

ESP32SPISlave slave;
WiFiServer WifiServer(12000);
WiFiClient client = WifiServer.available();

static constexpr uint32_t BUFFER_SIZE {32};
uint8_t spi_slave_tx_buf[BUFFER_SIZE];
uint8_t spi_slave_rx_buf[BUFFER_SIZE];

// Replace with your network credentials
const char* ssid = "ENTER_CURRENT_SSID";
const char* password = "ENTER_CURRENT_PSWD";

//Server we are sending the data to and connection timeout in ms
const char *ip = "13.41.53.180";
const uint port = 12000;
const int timeout = 3000;

//Time formatting using an NTP Server
const char* ntpServer = "pool.ntp.org";
const long  gmtOffset_sec = 0;
const int   daylightOffset_sec = 0;
char currentTime[26] = {'\0'};

//Initalise the Wifi Connection (can be re-ran for re-connection)
void initWiFi() {
  Serial.println("--------------[WIFI COMM]----------");
  WiFi.mode(WIFI_STA);
  Serial.print("[WiFi] | SSID:");
  Serial.print(ssid);
  Serial.print(" PSWD:");
  Serial.println(password);
  WiFi.begin(ssid, password);
  Serial.println("[WiFi] | Starting Connection");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] | Waiting for Connection");
    delay(5000);
  }
  Serial.print("[WiFi] | Connected at: ");
  Serial.println(WiFi.localIP());
  
}

void printLocalTime(){
  struct tm timeinfo;
  if(!getLocalTime(&timeinfo)){
    Serial.println("Failed to obtain time");
    return;
  }
  strftime(currentTime, 26, "%Y-%m-%d %H:%M:%S", &timeinfo);
  puts(currentTime);
  Serial.print("[NTP] | ");
  Serial.println(currentTime);
}

void setup() {
    Serial.begin(115200);
    Serial.println("====================[SETUP COMM]=================");
    //Connect to Wifi
    initWiFi();

    // Init and get the time
    configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
    printLocalTime();

    //Start SPI Connection with HSPI Pins
    //HSPI = CS: 15, CLK: 14, MOSI: 13, MISO: 12
    slave.setDataMode(SPI_MODE0);
    slave.begin(HSPI);

    WifiServer.begin();

}


void sendRequest(const char* Message) {

  Serial.println("--------------[TPC COMM]-----------");
  //Check if connected to the server as a client
  if(!client.connected()){

    if(client.connect(ip,port,timeout)){
      Serial.println("[TCP_IP]   | succesful! server connected");
    } 
    else {
      Serial.println("[TCP_IP]   | could not connect to server, timout reached");
    }

  }

  else{
    Serial.println("[TCP_IP]  | Open Connection Maintained");

    //Send the Data if connected
    client.print(Message); 
    Serial.print("[TCP_Tx] | ");
    Serial.println(Message);

    // read entire response untill hit newline char
    Serial.print("[TCP_Rx] | ");

    for(int i=0; i < 8; i++){
      char val = client.read();
      Serial.print(val);
    }
  }

}


void loop() {
    // block until the transaction comes from master
    slave.wait(spi_slave_rx_buf, spi_slave_tx_buf, BUFFER_SIZE);

    // if transaction has completed from master,
    // available() returns size of results of transaction,
    // and `spi_slave_rx_buf` is automatically updated
    while (slave.available()) {
        Serial.println("=====================================[TRANMISSION BLOCK]===================================");
        Serial.println("--------------[SPI COMM]-----------");

        //print the full Spi Rx Buffer
        Serial.print("[SPI_Rx]  | ");
        for(int i=0; i < 8; i++){
          int val = spi_slave_rx_buf[i];
          printf("%d | ", val);
        }
        printf("\n");

        //string format for json data with database
        char PostData[128];
        printLocalTime();
        sprintf(PostData, "{\"timestamp\":\"%s\", \"device_id\":\"1\", \"change_step\":\"%d\", \"heading\":\"%d\"}", currentTime, spi_slave_rx_buf[0], spi_slave_rx_buf[1]);
        Serial.print("[SPI->MSG] | ");
        Serial.println(PostData);

        //If disconnected Re-Connect 
        if (WiFi.status() != WL_CONNECTED){
          Serial.println("[WiFi] | Disconnected Re-Connecting");
          initWiFi();
        }
          

        //send the formatted string
        sendRequest(PostData);

        slave.pop();//Remove the read part of the Spi Rx Buffer
        slave.pop();
    }
}