//@version=5
indicator('JohnScript', format=format.price, precision=4, overlay=true)

// Inputs
a = input(1, title='Чувствительность')
c = input(10, title='Период ATR')
h = input(false, title='Сигналы Heikin Ashi')
signal_length = input.int(title='Сглаживание', minval=1, maxval=200, defval=11)
sma_signal = input(title='Сигнальная линия (MA)', defval=true)
lin_reg = input(title='Линейная регрессия', defval=false)
linreg_length = input.int(title='Длина линейной регрессии', minval=1, maxval=200, defval=11)

// Линии Болинджера
bollinger = input(false, title='Боллинджер')
bolingerlength = input(20, 'Длина')
// Bollinger Bands
bsrc = input(close, title='Исходные данные')
mult = input.float(2.0, title='Смещение', minval=0.001, maxval=50)
basis = ta.sma(bsrc, bolingerlength)
dev = mult * ta.stdev(bsrc, bolingerlength)
upper = basis + dev
lower = basis - dev
plot(bollinger ? basis : na, color=color.new(color.red, 0), title='Bol Basic')
p1 = plot(bollinger ? upper : na, color=color.new(color.blue, 0), title='Bol Upper')
p2 = plot(bollinger ? lower : na, color=color.new(color.blue, 0), title='Bol Lower')
fill(p1, p2, title='Bol Background', color=color.new(color.blue, 90))

// EMA
len = input(title='Длина EMA', defval=50)
//smooth = input (title="Сглаживание", type=input.bool, defval=false)
ema1 = ta.ema(close, len)
plot(ema1, color=color.new(color.yellow, 0), linewidth=2, title='EMA')


xATR = ta.atr(c)
nLoss = a * xATR

src = h ? request.security(ticker.heikinashi(syminfo.tickerid), timeframe.period, close, lookahead=barmerge.lookahead_off) : close

xATRTrailingStop = 0.0
iff_1 = src > nz(xATRTrailingStop[1], 0) ? src - nLoss : src + nLoss
iff_2 = src < nz(xATRTrailingStop[1], 0) and src[1] < nz(xATRTrailingStop[1], 0) ? math.min(nz(xATRTrailingStop[1]), src + nLoss) : iff_1
xATRTrailingStop := src > nz(xATRTrailingStop[1], 0) and src[1] > nz(xATRTrailingStop[1], 0) ? math.max(nz(xATRTrailingStop[1]), src - nLoss) : iff_2

pos = 0
iff_3 = src[1] > nz(xATRTrailingStop[1], 0) and src < nz(xATRTrailingStop[1], 0) ? -1 : nz(pos[1], 0)
pos := src[1] < nz(xATRTrailingStop[1], 0) and src > nz(xATRTrailingStop[1], 0) ? 1 : iff_3

xcolor = pos == -1 ? color.red : pos == 1 ? color.green : color.blue

ema = ta.ema(src, 1)
above = ta.crossover(ema, xATRTrailingStop)
below = ta.crossover(xATRTrailingStop, ema)

buy = src > xATRTrailingStop and above
sell = src < xATRTrailingStop and below

barbuy = src > xATRTrailingStop
barsell = src < xATRTrailingStop

plotshape(buy, title='Buy', text='Buy', style=shape.labelup, location=location.belowbar, color=color.new(color.green, 0), textcolor=color.new(color.white, 0), size=size.tiny)
plotshape(sell, title='Sell', text='Sell', style=shape.labeldown, location=location.abovebar, color=color.new(color.red, 0), textcolor=color.new(color.white, 0), size=size.tiny)

barcolor(barbuy ? color.green : na)
barcolor(barsell ? color.red : na)

alertcondition(buy, 'UT Long', 'UT Long')
alertcondition(sell, 'UT Short', 'UT Short')

bopen = lin_reg ? ta.linreg(open, linreg_length, 0) : open
bhigh = lin_reg ? ta.linreg(high, linreg_length, 0) : high
blow = lin_reg ? ta.linreg(low, linreg_length, 0) : low
bclose = lin_reg ? ta.linreg(close, linreg_length, 0) : close

r = bopen < bclose

signal = sma_signal ? ta.sma(bclose, signal_length) : ta.ema(bclose, signal_length)

plotcandle(r ? bopen : na, r ? bhigh : na, r ? blow : na, r ? bclose : na, title='LinReg Candles', color=color.green, wickcolor=color.green, bordercolor=color.green, editable=true)
plotcandle(r ? na : bopen, r ? na : bhigh, r ? na : blow, r ? na : bclose, title='LinReg Candles', color=color.red, wickcolor=color.red, bordercolor=color.red, editable=true)

plot(signal, color=color.new(color.white, 0))