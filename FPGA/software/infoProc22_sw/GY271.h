#include "i2c_opencores.h"
#include "i2c_opencores_regs.h"

#define I2C_REG 0xD
#define X_REG 0x0
#define Y_REG 0x2
#define Z_REG 0x4
#define TMP_REG 0x7

alt_32 base_gy271;

void GY_271_init(alt_32 _base, alt_32 clk, alt_32 speed){
    base_gy271 =  _base;
    I2C_init(base_gy271,clk,speed);
}

void GY_271_Write(alt_u8 reg,alt_u8 data){
    I2C_start(base_gy271,I2C_REG,0);
    I2C_write(base_gy271,reg,0);
    I2C_write(base_gy271,data,1);
}

alt_u8 GY_271_Read(alt_u8 reg){
    I2C_start(base_gy271,I2C_REG,0);
    I2C_write(base_gy271,reg,0);
    I2C_start(base_gy271,I2C_REG,1);
    return I2C_read(base_gy271,1);
}

alt_16 GY_271_Read_TMP(){
    alt_16 out = GY_271_Read(TMP_REG);
    out |= GY_271_Read(TMP_REG+1)<<8;
    return out;
}

alt_16 GY_271_Read_x(){
    alt_16 out = GY_271_Read(X_REG);
    out |= GY_271_Read(X_REG+1)<<8;
    return out;
}

alt_16 GY_271_Read_y(){
    alt_16 out = GY_271_Read(Y_REG);
    out |= GY_271_Read(Y_REG+1)<<8;
    return out;
}

alt_16 GY_271_Read_z(){
    alt_16 out = GY_271_Read(Z_REG);
    out |= GY_271_Read(Z_REG+1)<<8;
    return out;
}

void GY_271_Roll_Over_read(alt_16 * x,alt_16 * y,alt_16 * z, alt_u8 * r){
    I2C_start(base_gy271,I2C_REG,0);
    I2C_write(base_gy271,0,1);
    I2C_start(base_gy271,I2C_REG,1);
    *x = I2C_read(base_gy271,0);
    *x |= I2C_read(base_gy271,0)<<8;
    *y = I2C_read(base_gy271,0);
    *y |= I2C_read(base_gy271,0)<<8;
    *z = I2C_read(base_gy271,0);
    *z |= I2C_read(base_gy271,0)<<8;
    *r = I2C_read(base_gy271,1);
}

void GY_271_setMode(alt_u8 OSR, alt_u8 RNG, alt_u8 ODR, alt_u8 MODE, alt_u8 ROL_PNT, alt_u8 INT_ENB) {
    OSR &= 0b11;
    RNG &= 0b1;
    ODR &= 0b11;
    MODE &= 0b1;
    ROL_PNT &= 0b1;
    INT_ENB &= 0b1;
    alt_u8 regVal = (OSR << 6) | (RNG << 4) | (ODR << 2) | MODE;
    GY_271_Write(0x9, regVal);
    regVal = (ROL_PNT << 6) | INT_ENB;
    GY_271_Write(0xA, regVal); 
    GY_271_Write(0xB,0x1);
}

void GY_271_Reset(){
    GY_271_Write(0xA,0x8);
    GY_271_Write(0xA,0x0);
}