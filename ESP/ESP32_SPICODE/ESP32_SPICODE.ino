#include <ESP32SPISlave.h>
#include <WiFi.h>
#include <time.h>
#include <WString.h>

ESP32SPISlave slave;
WiFiServer WifiServer(12000);
WiFiClient client = WifiServer.available();

static constexpr uint32_t BUFFER_SIZE {2};
uint8_t spi_slave_tx_buf[BUFFER_SIZE];
uint8_t spi_slave_rx_buf[BUFFER_SIZE];

// Replace with your network credentials
const char* ssid = "Diego-XPS";
const char* password = "helloGitHub!";

//Server we are sending the data to and connection timeout in ms
const char *ip = "13.41.53.180";
const uint port = 12000;
const int timeout = 3000;

//Time formatting using an NTP Server
const char* ntpServer = "0.uk.pool.ntp.org";
const long  gmtOffset_sec = 0;
const int   daylightOffset_sec = 0;
char currentTime[26] = {'\0'};

//Total Variables
long int total_steps;

//Initalise the Wifi Connection (can be re-ran for re-connection)
void initWiFi() {

  Serial.println("[WiFi] | Checking the Current Connection");

  if(WL_CONNECTED != WiFi.status()){

    Serial.println("--------------[WIFI COMM]----------");
    Serial.println("[WiFi] | Disconnected From Wifi");
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

  }else {

    Serial.println("[WiFi] | WiFi is Connected");

  }
  
}

bool printLocalTime(){

  initWiFi();

  struct tm timeinfo;
  if(!getLocalTime(&timeinfo)){
    Serial.println("[NTP] | Failed to obtain time");
    return false;
  }

  strftime(currentTime, 26, "%FT%TZ", &timeinfo);
  Serial.print("[NTP] | ");
  Serial.println(currentTime);
  return true;


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

  total_steps = 0;

  WifiServer.begin();

  sendRequest("Cozzy");
}


void sendRequest(const char* Message) {
  Serial.println("--------------[TPC COMM]-----------");

  initWiFi();

  //Check if connected to the server as a client
  if(!client.connected()){

      if(client.connect(ip,port,timeout)){

        Serial.println("[TCP_IP]   | successful! server connected");
        
        //Send the Data if connected
        client.println(Message); 
        Serial.print("[TCP_Tx] | ");
        Serial.println(Message);

    } 
    else {

      Serial.println("[TCP_IP]   | could not connect to server, timeout reached");
    
    }

  }

  else{
    Serial.println("[TCP_IP]  | Open Connection Maintained");

    //Send the Data if connected
    client.println(Message); 
    Serial.print("[TCP_Tx] | ");
    Serial.println(Message);

    // read entire response untill hit newline char
    Serial.print("[TCP_Rx] | ");

    while (!client.available());                // wait for response

    String val;
    val = client.readStringUntil('\n');
    Serial.print(val);
    Serial.println();
  }

  // client.stop();

}


void loop() {

    // block until the transaction comes from master
    slave.wait(spi_slave_rx_buf, spi_slave_tx_buf, BUFFER_SIZE);

    // if transaction has completed from master,
    // available() returns size of results of transaction,
    // and `spi_slave_rx_buf` is automatically updated
    while (slave.available() > 1) {
        Serial.println("=====================================[TRANSMISSION BLOCK]===================================");
        Serial.println("--------------[SPI COMM]-----------");

        //print the full Spi Rx Buffer
        Serial.print("[SPI_Rx]  | ");
        for(int i=0; i < 2; i++){
          int val = spi_slave_rx_buf[i];
          printf("%d | ", val);
        }
        printf("\n");

        //print the full Spi Rx Buffer
        Serial.print("[SPI_Tx]  | ");
        for(int i=0; i < 2; i++){
          spi_slave_tx_buf[i] = 65;
          printf("%d | ", spi_slave_tx_buf[i]);
        }
        printf("\n");

        //update the current time
        if(printLocalTime()){
          //buffer for the Json Data
          char PostData[128];

          //Format the Json Data
          sprintf(PostData, "{\"timestamp\":\"%s\", \"device_id\":\"Cozzy\", \"total_steps\":%d, \"heading\":%d}", currentTime, spi_slave_rx_buf[0], spi_slave_rx_buf[1]);
          Serial.print("[SPI->MSG] | ");
          Serial.println(PostData);

          //send the formatted string
          sendRequest(PostData);

          slave.pop();//Remove the read part of the Spi Rx Buffer
          slave.pop();

        }

    }
}
