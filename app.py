import os
from flask import Flask, render_template
import threading
import time
import ccxt
import pandas as pd
import ta
import requests
import random
import string
import hashlib
import json
import numpy as np
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# Configuración del estado del bot
bot_status = {
    "running": False,
    "last_activity": None,
    "signals": [],
    "errors": [],
    "start_time": None
}

# Configuraciones globales
MIN_VOLUME_USDT = 500000
MAX_SPREAD_PERCENT = 0.15
CAPITAL_TOTAL_USD = 1000
RIESGO_POR_OPERACION_USD = 5
LEVERAGE = 20
SLEEP_MINUTES = 5

# Binance (para análisis)
exchange = ccxt.binance({'enableRateLimit': True})
exchange.load_markets()

# Telegram
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '8494135071:AAGqUhnA4v0GBa3OH3PvdzrFd-1Pr-BGpZM')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '1104103584')

# Bitunix API - Cuenta principal
BITUNIX_API_KEY_1 = os.getenv('BITUNIX_API_KEY_1', 'c057a91c15ca435af71ffb7456438751')
BITUNIX_API_SECRET_1 = os.getenv('BITUNIX_API_SECRET_1', '64a16d7d8fb48a3e2aa50756fc154ac8')

# Bitunix API - Cuenta secundaria
BITUNIX_API_KEY_2 = os.getenv('BITUNIX_API_KEY_2', '4e351cc3e6487e0a87c77e304b9daac6')
BITUNIX_API_SECRET_2 = os.getenv('BITUNIX_API_SECRET_2', 'accadcd841560463a821fe450e83fa7d')

def enviar_telegram(mensaje):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": mensaje}
    try:
        requests.post(url, data=data)
    except Exception as e:
        print("Error Telegram:", e)
        bot_status["errors"].append(f"Error Telegram: {str(e)}")

def obtener_datos(symbol, timeframe='15m', limit=150):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Error obteniendo datos {symbol}: {e}")
        bot_status["errors"].append(f"Error obteniendo datos {symbol}: {str(e)}")
        return None

def calcular_choppiness_index(df, period=14):
    atr = df['high'].rolling(period).max() - df['low'].rolling(period).min()
    high_low_std = df['close'].rolling(period).std()
    chop = 100 * np.log10(atr / high_low_std) / np.log10(period)
    return chop

def calcular_indicadores(df):
    df['ema_fast'] = EMAIndicator(df['close'], window=9).ema_indicator()
    df['ema_slow'] = EMAIndicator(df['close'], window=21).ema_indicator()
    df['adx'] = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['rsi'] = RSIIndicator(df['close']).rsi()
    df['choppiness'] = calcular_choppiness_index(df)
    return df

def detectar_patrones(df):
    ultima = df.iloc[-1]
    previa = df.iloc[-2]
    cruce_ema_bull = previa['ema_fast'] < previa['ema_slow'] and ultima['ema_fast'] > ultima['ema_slow']
    cruce_ema_bear = previa['ema_fast'] > previa['ema_slow'] and ultima['ema_fast'] < ultima['ema_slow']
    return cruce_ema_bull, cruce_ema_bear

def detectar_divergencia(df):
    macd_hist = df['macd'] - df['macd_signal']
    rsi = df['rsi']
    if macd_hist.iloc[-1] > macd_hist.iloc[-2] and rsi.iloc[-1] < rsi.iloc[-2]:
        return 'posible divergencia bajista'
    elif macd_hist.iloc[-1] < macd_hist.iloc[-2] and rsi.iloc[-1] > rsi.iloc[-2]:
        return 'posible divergencia alcista'
    return None

def detectar_soportes_resistencias(df, n=3, margen_pct=1.5):
    soportes = []
    resistencias = []
    for i in range(n, len(df) - n):
        minimos_anteriores = all(df['low'].iloc[i] < df['low'].iloc[i - j] for j in range(1, n + 1))
        minimos_posteriores = all(df['low'].iloc[i] < df['low'].iloc[i + j] for j in range(1, n + 1))
        maximos_anteriores = all(df['high'].iloc[i] > df['high'].iloc[i - j] for j in range(1, n + 1))
        maximos_posteriores = all(df['high'].iloc[i] > df['high'].iloc[i + j] for j in range(1, n + 1))

        if minimos_anteriores and minimos_posteriores:
            soporte = df['low'].iloc[i]
            if not any(abs(soporte - s) / soporte < margen_pct / 100 for s in soportes):
                soportes.append(soporte)

        if maximos_anteriores and maximos_posteriores:
            resistencia = df['high'].iloc[i]
            if not any(abs(resistencia - r) / resistencia < margen_pct / 100 for r in resistencias):
                resistencias.append(resistencia)

    return soportes, resistencias

def detectar_price_action(df):
    candle = df.iloc[-1]
    body = abs(candle['close'] - candle['open'])
    candle_range = candle['high'] - candle['low']
    upper_shadow = candle['high'] - max(candle['close'], candle['open'])
    lower_shadow = min(candle['close'], candle['open']) - candle['low']

    # Doji
    if body < candle_range * 0.1:
        return "Doji"
    # Hammer
    if lower_shadow > body * 2 and upper_shadow < body * 0.5:
        return "Hammer"
    # Shooting Star
    if upper_shadow > body * 2 and lower_shadow < body * 0.5:
        return "Shooting Star"
    # Bullish Engulfing (simplificado)
    previa = df.iloc[-2]
    if (candle['close'] > candle['open'] and previa['close'] < previa['open'] and
            candle['open'] < previa['close'] and candle['close'] > previa['open']):
        return "Bullish Engulfing"
    # Bearish Engulfing (simplificado)
    if (candle['close'] < candle['open'] and previa['close'] > previa['open'] and
            candle['open'] > previa['close'] and candle['close'] < previa['open']):
        return "Bearish Engulfing"

    return None

def validar_multiframe(symbol):
    df_15m = obtener_datos(symbol, '15m')
    df_1h = obtener_datos(symbol, '1h')
    if df_15m is None or df_1h is None:
        return None, None, None

    df_15m = calcular_indicadores(df_15m)
    df_1h = calcular_indicadores(df_1h)

    # Filtro de mercado lateral (choppiness)
    if df_15m['choppiness'].iloc[-1] > 60 or df_1h['choppiness'].iloc[-1] > 60:
        print(f"[{symbol}] Mercado lateral (Choppiness > 60), se omite.")
        return None, None, None

    b15, s15 = detectar_patrones(df_15m)
    b1h, s1h = detectar_patrones(df_1h)
    div15 = detectar_divergencia(df_15m)
    div1h = detectar_divergencia(df_1h)
    patron_vela = detectar_price_action(df_15m)

    precio_actual = df_15m['close'].iloc[-1]

    # Detectar soportes y resistencias
    soportes, resistencias = detectar_soportes_resistencias(df_15m)
    print(f"[{symbol}] 🟦 Soportes detectados: {[round(s, 5) for s in soportes]}")
    print(f"[{symbol}] 🟥 Resistencias detectadas: {[round(r, 5) for r in resistencias]}")
    print(f"[{symbol}] Precio actual: {precio_actual:.5f}")

    margen = 0.01  # 1% de margen de proximidad

    cerca_resistencia = any(abs(precio_actual - r) / precio_actual < margen for r in resistencias)
    cerca_soporte = any(abs(precio_actual - s) / precio_actual < margen for s in soportes)

    # Filtrar señales según patrón de vela:
    if patron_vela:
        print(f"[{symbol}] Patrón de vela detectado: {patron_vela}")
        # Si patrón bajista y señal LONG, descartar
        if patron_vela in ["Shooting Star", "Bearish Engulfing", "Doji"] and b15 and b1h:
            print(f"⛔ [{symbol}] Señal LONG evitada por patrón bajista: {patron_vela}")
            return None, df_15m, None
        # Si patrón alcista y señal SHORT, descartar
        if patron_vela in ["Hammer", "Bullish Engulfing"] and s15 and s1h:
            print(f"⛔ [{symbol}] Señal SHORT evitada por patrón alcista: {patron_vela}")
            return None, df_15m, None

    if b15 and b1h:
        if cerca_resistencia:
            print(f"⛔ [{symbol}] Señal LONG evitada por resistencia cercana.")
            return None, df_15m, None
        return 'LONG', df_15m, div15 or div1h
    elif s15 and s1h:
        if cerca_soporte:
            print(f"⛔ [{symbol}] Señal SHORT evitada por soporte cercano.")
            return None, df_15m, None
        return 'SHORT', df_15m, div15 or div1h

    return None, df_15m, None

def calcular_stop_loss_take_profit(df, direccion):
    rango = df['high'].iloc[-14:].max() - df['low'].iloc[-14:].min()
    precio = df['close'].iloc[-1]
    if direccion == 'LONG':
        sl = precio - rango * 0.5
        tp = precio + rango * 1.5
    else:
        sl = precio + rango * 0.5
        tp = precio - rango * 1.5
    return round(precio, 4), round(sl, 4), round(tp, 4)

def calcular_tamano_posicion_fijo(precio_entrada):
    monto_total = RIESGO_POR_OPERACION_USD * LEVERAGE
    cantidad = monto_total / precio_entrada
    return round(cantidad, 3)

def calcular_spread_percent(symbol):
    try:
        ob = exchange.fetch_order_book(symbol)
        bid = ob['bids'][0][0] if ob['bids'] else None
        ask = ob['asks'][0][0] if ob['asks'] else None
        if bid is None or ask is None:
            return None
        return round(((ask - bid) / ask) * 100, 4)
    except Exception as e:
        print(f"Error calcular spread {symbol}: {e}")
        bot_status["errors"].append(f"Error calcular spread {symbol}: {str(e)}")
        return None

def generar_nonce():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=16))

def sha256_hex(input_string):
    return hashlib.sha256(input_string.encode('utf-8')).hexdigest()

def generar_firma_bitunix(nonce, timestamp, api_key, secret_key, body_dict):
    query_params = ""
    body_str = json.dumps(body_dict, separators=(',', ':'))
    digest_input = nonce + timestamp + api_key + query_params + body_str
    digest = sha256_hex(digest_input)
    sign_input = digest + secret_key
    sign = sha256_hex(sign_input)
    return sign

def ejecutar_orden_bitunix(symbol, side, qty, leverage, entry_price, sl_price, tp_price,
                          api_key, api_secret, position_id=0):
    url = "https://fapi.bitunix.com/api/v1/futures/trade/place_order"
    nonce = generar_nonce()
    timestamp = str(int(time.time() * 1000))
    client_id = str(int(time.time() * 1000))
    side_str = "BUY" if side == "buy" else "SELL"
    trade_side = "OPEN"
    order_type = "LIMIT"
    reduce_only = False
    effect = "GTC"
    symbol_clean = symbol.replace("/", "").replace(":USDT", "")

    body_params = {
        "symbol": symbol_clean,
        "side": side_str,
        "price": str(round(entry_price, 4)),
        "qty": str(qty),
        "positionId": str(position_id) if position_id != 0 else "",
        "tradeSide": trade_side,
        "orderType": order_type,
        "reduceOnly": reduce_only,
        "effect": effect,
        "clientId": client_id,
        "tpPrice": str(round(tp_price, 4)),
        "tpStopType": "MARK",
        "tpOrderType": "LIMIT",
        "tpOrderPrice": str(round(tp_price * 1.0001, 4)),
        "slPrice": str(round(sl_price, 4)),
        "slStopType": "MARK",
        "slOrderType": "LIMIT",
        "slOrderPrice": str(round(sl_price * 0.9999 if side == "buy" else sl_price * 1.0001, 4))
    }
    body_params = {k: v for k, v in body_params.items() if v != ""}
    body_str = json.dumps(body_params, separators=(',', ':'))
    sign = generar_firma_bitunix(nonce, timestamp, api_key, api_secret, body_params)

    headers = {
        "api-key": api_key,
        "sign": sign,
        "nonce": nonce,
        "timestamp": timestamp,
        "language": "en-US",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, headers=headers, data=body_str)

        if response.status_code != 200:
            error_msg = f"Error HTTP {response.status_code} - {response.text}"
            print(error_msg)
            bot_status["errors"].append(error_msg)
            return None
        
        data = response.json()
        print(f"✅ Orden enviada Bitunix ({api_key[:6]}...):", data)
        return data
    except Exception as e:
        error_msg = f"Error ejecutando orden en Bitunix: {str(e)}"
        print(error_msg)
        bot_status["errors"].append(error_msg)
        return None

def modo_diagnostico():
    print("🔍 Ejecutando modo diagnóstico...\n")
    diagnostic_results = []

    # 1. Verificar conexión a Binance
    try:
        test_ticker = exchange.fetch_ticker("BTC/USDT")
        msg = f"✅ Binance conectado. Precio BTC: {test_ticker['last']}"
        print(msg)
        diagnostic_results.append(msg)
    except Exception as e:
        msg = f"❌ Error conexión Binance: {e}"
        print(msg)
        diagnostic_results.append(msg)
        return False, diagnostic_results

    # 2. Verificar conexión a Bitunix para ambas cuentas
    for i, (api_key, api_secret) in enumerate([(BITUNIX_API_KEY_1, BITUNIX_API_SECRET_1),
                                             (BITUNIX_API_KEY_2, BITUNIX_API_SECRET_2)], start=1):
        try:
            nonce = generar_nonce()
            timestamp = str(int(time.time() * 1000))
            margin_coin = "USDT"
            query_string = f"?marginCoin={margin_coin}"
            body_dict = {}  # GET sin body

            sign = generar_firma_bitunix(nonce, timestamp, api_key, api_secret, body_dict)

            headers = {
                "api-key": api_key,
                "sign": sign,
                "nonce": nonce,
                "timestamp": timestamp,
                "language": "en-US",
                "Content-Type": "application/json"
            }

            url = f"https://fapi.bitunix.com/api/v1/futures/account{query_string}"
            resp = requests.get(url, headers=headers)
            if resp.status_code == 200:
                msg = f"✅ Bitunix cuenta {i} conectado correctamente."
                print(msg)
                diagnostic_results.append(msg)
            else:
                msg = f"❌ Error conexión Bitunix cuenta {i}: {resp.status_code} - {resp.text}"
                print(msg)
                diagnostic_results.append(msg)
                return False, diagnostic_results
        except Exception as e:
            msg = f"❌ Error conexión Bitunix cuenta {i}: {e}"
            print(msg)
            diagnostic_results.append(msg)
            return False, diagnostic_results

    # 3. Verificar envío a Telegram
    try:
        mensaje_test = "✅ Prueba de conexión: Bot conectado correctamente."
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": mensaje_test}
        resp = requests.post(url, data=data)
        if resp.status_code == 200:
            msg = "✅ Telegram conectado correctamente."
            print(msg)
            diagnostic_results.append(msg)
        else:
            msg = f"❌ Error conexión Telegram: {resp.status_code} - {resp.text}"
            print(msg)
            diagnostic_results.append(msg)
            return False, diagnostic_results
    except Exception as e:
        msg = f"❌ Error Telegram: {e}"
        print(msg)
        diagnostic_results.append(msg)
        return False, diagnostic_results

    print("\n✅ Todas las conexiones verificadas. Iniciando bot...\n")
    return True, diagnostic_results

def bot_loop():
    bot_status["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    while bot_status["running"]:
        try:
            symbols = [s for s, m in exchange.markets.items() if m['quote'] == 'USDT' and m.get('contract', False)]
            for symbol in symbols:
                try:
                    ticker = exchange.fetch_ticker(symbol)
                    vol = ticker.get('quoteVolume') or 0
                    if vol < MIN_VOLUME_USDT:
                        print(f"[{symbol}] Volumen bajo: {vol}")
                        continue
                    
                    spread = calcular_spread_percent(symbol)
                    if spread is None or spread > MAX_SPREAD_PERCENT:
                        print(f"[{symbol}] Spread alto o no disponible: {spread}")
                        continue
                    
                    # Analizar y ejecutar
                    direccion, df, divergencia = validar_multiframe(symbol)
                    if direccion:
                        precio, sl, tp = calcular_stop_loss_take_profit(df, direccion)
                        cantidad = calcular_tamano_posicion_fijo(precio)
                        if cantidad == 0:
                            continue
                        
                        signal_info = {
                            "symbol": symbol,
                            "direction": direccion,
                            "entry": precio,
                            "sl": sl,
                            "tp": tp,
                            "size": cantidad,
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "divergence": divergencia
                        }
                        bot_status["signals"].append(signal_info)
                        
                        mensaje = (f"📊 Señal detectada en {symbol}\nTipo: {direccion}\n📥 Entrada: {precio}\n"
                                 f"🛑 SL: {sl}\n🎯 TP: {tp}\n💰 Tamaño: {cantidad} USD")
                        if divergencia:
                            mensaje += f"\n🔍 {divergencia}"
                        enviar_telegram(mensaje)
                        
                        # Ejecutar en cuenta 1
                        ejecutar_orden_bitunix(symbol,
                                            'buy' if direccion == 'LONG' else 'sell',
                                            cantidad,
                                            LEVERAGE,
                                            precio,
                                            sl,
                                            tp,
                                            BITUNIX_API_KEY_1,
                                            BITUNIX_API_SECRET_1)
                        
                        # Ejecutar en cuenta 2
                        ejecutar_orden_bitunix(symbol,
                                            'buy' if direccion == 'LONG' else 'sell',
                                            cantidad,
                                            LEVERAGE,
                                            precio,
                                            sl,
                                            tp,
                                            BITUNIX_API_KEY_2,
                                            BITUNIX_API_SECRET_2)
                except Exception as e:
                    error_msg = f"Error en {symbol}: {str(e)}"
                    bot_status["errors"].append(error_msg)
                    print(error_msg)
            
            bot_status["last_activity"] = time.strftime("%Y-%m-%d %H:%M:%S")
            time.sleep(SLEEP_MINUTES * 60)
            
        except Exception as e:
            error_msg = f"Error en el bucle principal: {str(e)}"
            bot_status["errors"].append(error_msg)
            print(error_msg)
            time.sleep(60)  # Esperar antes de reintentar

@app.route('/')
def dashboard():
    return render_template('dashboard.html', 
                         status=bot_status,
                         signals=bot_status["signals"][-10:][::-1],  # Últimas 10 señales
                         errors=bot_status["errors"][-10:][::-1],    # Últimos 10 errores
                         uptime=time.strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/start')
def start_bot():
    if not bot_status["running"]:
        success, diagnostics = modo_diagnostico()
        if success:
            bot_status["running"] = True
            threading.Thread(target=bot_loop, daemon=True).start()
            return "Bot iniciado correctamente"
        else:
            return f"Error en diagnóstico:<br>" + "<br>".join(diagnostics)
    return "El bot ya está en ejecución"

@app.route('/stop')
def stop_bot():
    bot_status["running"] = False
    return "Bot detenido"

@app.route('/status')
def get_status():
    return {
        "running": bot_status["running"],
        "last_activity": bot_status["last_activity"],
        "signal_count": len(bot_status["signals"]),
        "error_count": len(bot_status["errors"]),
        "start_time": bot_status["start_time"],
        "uptime": time.strftime("%Y-%m-%d %H:%M:%S")
    }

@app.route('/signals')
def get_signals():
    return {
        "signals": bot_status["signals"][-10:][::-1],
        "count": len(bot_status["signals"])
    }

@app.route('/')
def home():
    try:
        logger.info("Solicitud recibida en la ruta principal")
        return render_template('dashboard.html')
    except Exception as e:
        logger.error(f"Error al renderizar plantilla: {str(e)}")
        return f"Error en la aplicación: {str(e)}", 500

def create_app():
    return app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    logger.info(f"Iniciando aplicación en el puerto {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
    app.run(host='0.0.0.0', port=port)


