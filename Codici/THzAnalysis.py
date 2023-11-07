import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cnst
import warnings as wn

import lmfit as fit

import src.data_manipulation_functions as man
import src.math_functions as mt
import ConductivityModels as mod

# plt.style.use('ggplot')


class THzAnalyser(object):
        """docstring for THzAnalyser"""
        def __init__(self, fileList, refList,
                     fmt, sampleName, d=0, ns=0, n2=1,
                     exp=0, start=0, stop=0, flip=False, plot=True,
                     window='',
                     fitFlag=False, model='', complexFunction=False,
                     init=[], para=[], boundaries=[0, 1000], guess=False,
                     fitQty='Conductivity'):
                super(THzAnalyser, self).__init__()
                self.fileList = fileList
                self.window = window
                self.para = para
                self.sample = mod.Sample(sampleName, d=d, ns=ns, n2=n2)
                self.params = 0

                if fmt == 'Ox':
                        sigCol = 2
                        refCol = 1
                elif fmt == 'TW':
                        sigCol = 1
                        refCol = 7
                elif fmt == 'abcd':
                        sigCol = 1
                        refCol = 2
                else:
                        sigCol = 1
                        refCol = 1

                shapeData = self.Data_Reader(fileList[0], fmt, 0, shape=0)
                shape = np.shape(shapeData)
                lenFiles = shape[1]
                numFiles = len(fileList)
                lenfft = np.int(np.round((exp + 1) *
                                         (lenFiles -
                                          stop - start) / 2 + .6))
                listShape = ((numFiles,
                              (exp + 1) * (lenFiles -
                                           stop - start)))
                lenT = listShape[1]
                listShapeFft = ((numFiles, lenfft))

                self.xList = np.zeros(listShape)
                self.xRefList = np.zeros(listShape)

                self.tList = np.zeros(listShape)
                self.tRefList = np.zeros(listShape)

                self.EtList = np.zeros(listShape)
                self.EtRefList = np.zeros(listShape)

                self.t = np.zeros(lenT)
                self.Et = np.zeros(lenT)
                self.EtRef = np.zeros(lenT)

                self.fList = np.zeros(listShapeFft,
                                      dtype=np.complex_)
                self.EList = np.zeros(listShapeFft,
                                      dtype=np.complex_)
                self.ERefList = np.zeros(listShapeFft,
                                         dtype=np.complex_)
                self.transList = np.zeros(listShapeFft,
                                          dtype=np.complex_)
                self.sigmaList = np.zeros(listShapeFft,
                                          dtype=np.complex_)
                self.epsilonList = np.zeros(listShapeFft,
                                            dtype=np.complex_)
                self.lossList = np.zeros(listShapeFft,
                                         dtype=np.complex_)
                self.nList = np.zeros(listShapeFft,
                                      dtype=np.complex_)

                self.f = np.zeros(lenfft,
                                  dtype=np.complex_)
                self.E = np.zeros(lenfft,
                                  dtype=np.complex_)
                self.ERef = np.zeros(lenfft,
                                     dtype=np.complex_)
                self.sigma = np.zeros(lenfft,
                                      dtype=np.complex_)
                self.trans = np.zeros(lenfft,
                                      dtype=np.complex_)
                self.epsilon = np.zeros(lenfft,
                                        dtype=np.complex_)
                self.loss = np.zeros(lenfft,
                                     dtype=np.complex_)
                self.n = np.zeros(lenfft,
                                  dtype=np.complex_)

                self.sigmaUncReal = np.zeros(lenfft)
                self.sigmaUncImag = np.zeros(lenfft)
                self.transUnc = np.zeros(lenfft)
                self.epsilonUncReal = np.zeros(lenfft)
                self.epsilonUncImag = np.zeros(lenfft)
                self.lossUnc = np.zeros(lenfft)
                self.nUnc = np.zeros(lenfft)

                # self.DrudeCoeff = 0
                stop = lenFiles - stop
                if stop == 0:
                        stop = int(1e6)

                tmp_xList, tmp_EtList = self.Data_Reader(fileList, fmt,
                                                         sigCol, shape)
                tmp_xRefList, tmp_EtRefList = self.Data_Reader(refList, fmt,
                                                               refCol, shape)
                if np.abs(len(refList) - len(fileList)) != 0:
                        for i in range(len(fileList)):
                                tmp_xRefList[i] = tmp_xRefList[0]
                                tmp_EtRefList[i] = tmp_EtRefList[0]

                for i, file in enumerate(fileList):
                        (self.tList[i],
                         self.EtList[i],
                         self.tRefList[i],
                         self.EtRefList[i],
                         self.fList[i],
                         self.EList[i],
                         self.ERefList[i],
                         self.transList[i],
                         self.sigmaList[i],
                         self.epsilonList[i],
                         self.lossList[i],
                         self.nList[i]
                         ) = self.Data_Computation(
                             tmp_EtList[i, start:stop],
                             tmp_EtRefList[i, start:stop],
                             tmp_xList[i, start:stop],
                             tmp_xRefList[i, start:stop],
                             self.sample,
                             fmt,
                             flip=flip,
                             exp=exp,
                             window=window,
                             para=para)
                for i in range(lenT):
                        self.t[i] = np.average(self.tList[:, i])
                        self.Et[i] = np.average(self.EtList[:, i])
                        self.EtRef[i] = np.average(self.EtRefList[:, i])
                for i in range(lenfft):
                        self.f[i] = np.average(self.fList[:, i])
                        self.E[i] = np.average(self.EList[:, i])
                        self.ERef[i] = np.average(self.ERefList[:, i])
                        self.trans[i] = np.average(self.transList[:, i])
                        self.sigma[i] = np.average(self.sigmaList[:, i])
                        self.epsilon[i] = np.average(self.epsilonList[:, i])
                        self.loss[i] = np.average(self.lossList[:, i])
                        self.n[i] = np.average(self.nList[:, i])

                        self.sigmaUncReal[i] = np.std(
                            np.real(self.sigmaList[:, i]))
                        self.sigmaUncImag[i] = np.std(
                            np.imag(self.sigmaList[:, i]))
                        self.transUnc[i] = np.std(np.abs(self.transList[:, i]))
                        self.epsilonUncReal[i] = np.std(
                            np.real(self.epsilonList[:, i]))
                        self.epsilonUncImag[i] = np.std(
                            np.imag(self.epsilonList[:, i]))
                        self.lossUnc[i] = np.std(np.abs(self.lossList[:, i]))
                        self.nUnc[i] = np.std(np.abs(self.nList[:, i]))
                # for i in range(3, lenfft):
                #         tmp = np.sqrt(np.real(self.sigma[i])**2) \
                #             / np.sqrt(np.real(self.sigma[i])**2 +
                #                       np.imag(self.sigma[i])**2)
                #         self.DrudeCoeff += tmp
                #         if np.abs(self.f[i] - 2.5) < 0.03:
                #                 self.DrudeCoeff /= (i - 3)
                #                 break
                self.ratio = 1e15 * np.imag(self.sigma) / (np.abs(self.f) *
                                                           2e12 * np.pi *
                                                           np.real(self.sigma))
                if fitFlag:
                        y_Map = {'Conductivity': self.sigma,
                                 'Transmission': self.trans}
                        y = y_Map[fitQty]
                        err_Map = {'Conductivity':
                                   [self.sigmaUncReal[boundaries[0]:
                                                      boundaries[1]],
                                    self.sigmaUncImag[boundaries[0]:
                                                      boundaries[1]]],
                                   'Transmission':
                                   self.transUnc[boundaries[0]:
                                                 boundaries[1]]}
                        err = err_Map[fitQty]
                        if model == '':
                                model = self.sample.f
                                wn.warn('Warning:: model undefined, sample' +
                                        '\'s default chosen: ' +
                                        model, RuntimeWarning)
                        self.params = self.Fit(x=self.f[boundaries[0]:
                                                        boundaries[1]],
                                               y=y[boundaries[0]:
                                                   boundaries[1]],
                                               model=model,
                                               err=err,
                                               init=init, para=para,
                                               c=complexFunction, guess=guess,
                                               plot=plot,
                                               fitQty=fitQty)
                if plot:
                        (self.multifig,
                         self.multitimefig,
                         self.finalfig) = self.Data_Plotter(fmt)

                        self.valuesfig = 0

        def __str__(self):
                return (str(self.fileList) + ' ' +
                        str(self.sample) + ' ' +
                        str(self.params))

        @staticmethod
        def Data_Reader(fileList, fmt, col=0, shape=0):
                sr = 0
                delm = '\t'

                if fmt == 'TW':
                        sr = 3
                        delm = ','

                if shape == 0:
                        data = man.Reader(fileList,
                                          delimiter=delm,
                                          skipRows=sr,
                                          caller='')
                        return data
                elif shape != 0:
                        data = np.zeros((len(fileList), shape[0], shape[1]))

                        for i, file in enumerate(fileList):
                                data[i] = man.Reader(file,
                                                     delimiter=delm,
                                                     skipRows=sr,
                                                     caller='')

                        return data[:, 0], data[:, col]

        @staticmethod
        def Data_Computation(E, ERef, x, xRef,
                             sample, fmt, flip=False, exp=0,
                             window='', para=[]):
                vacImp = cnst.physical_constants['characteristic impedance' +
                                                 ' of vacuum']
                Z0 = vacImp[0]  # the value of z0
                e0 = cnst.epsilon_0
                u = 'mm'
                cv = 0.1499
                ns = sample.ns
                n2 = sample.n2
                d = sample.d
                eInf = sample.eInf
                M = np.amax(ERef)
                idx = np.where(ERef == M)[0][0]
                shift = x[idx]
                x -= shift
                xRef -= shift
                if flip:
                        E = np.flipud(E)
                        ERef = np.flipud(ERef)
                        # x = np.flipud(x)
                        # xRef = np.flipud(xRef)

                if fmt == 'Ox':
                        (ERef, E) = (ERef - 0.5 * E, ERef + 0.5 * E)
                        x = x - 24
                        xRef = xRef - 24
                elif fmt == 'Wa':
                        E = ERef - E
                        ERef = ERef
                elif fmt == 'TW':
                        E = E
                        ERef = ERef
                        u = 'OD'
                        cv = 0.2998
                elif fmt == 'abcd':
                        pass
                else:
                        pass
                        wn.warn('Warning:: undefined or wrong format, ' +
                                'default one chosen: abcd',
                                RuntimeWarning)
                if window != '':
                        x, E, = man.Window(window,
                                           [p * cv for p in para], x, E)
                        xRef, ERef, = man.Window(window,
                                                 [para[0] * cv, para[1] * cv],
                                                 xRef, ERef)
                if exp > 0:
                        x, E = mt.zeropad(x, E, exp)
                        xRef, ERef = mt.zeropad(xRef, ERef, exp)

                freq, Efft = mt.IFFT(x, E, u)
                freq, EReffft = mt.IFFT(xRef, ERef, u)

                t = x / cv
                tRef = xRef / cv

                trans = Efft / EReffft

                sigma = -(ns + n2) * (Efft - EReffft) / (Z0 * d * EReffft)

                epsilon = eInf + 1j * sigma / (freq * 2e12 * np.pi * e0)

                loss = np.imag(-1 / epsilon)

                n = np.sqrt(epsilon)
                return (t, E, tRef, ERef,
                        freq, Efft, EReffft,
                        trans, sigma, epsilon, loss, n)

        def Data_Plotter(self, fmt):
                fMin = 0.2
                fMax = 2
                fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2,
                                                             sharex='all',
                                                             squeeze=True)
                for s in self.sigmaList:
                        i = np.where(s == self.sigmaList)[0][0]
                        (idxMin,
                         idxMax,
                         reCMin,
                         reCMax) = man.Extrema(fMin, fMax,
                                               self.fList[i],
                                               np.array(
                                                   (np.real(
                                                       s),)))
                        (idxMin,
                         idxMax,
                         imCMin,
                         imCMax) = man.Extrema(fMin, fMax,
                                               self.fList[i],
                                               np.array((np.imag(s),)))
                        plot0 = ax0.plot(np.real(self.fList[i]),
                                         np.real(s), ls='-')
                        color = plot0[0].get_color()
                        ax0.plot(np.real(self.fList[i]),
                                 np.imag(s), ls='--', color=color)
                        ax0.set_ylim(min(reCMin, imCMin), max(reCMax, imCMax))

                        (idxMin,
                         idxMax,
                         reCMin,
                         reCMax) = man.Extrema(fMin, fMax,
                                               self.fList[i],
                                               np.array(
                                                   (np.real(
                                                       self.epsilonList[i]),)))
                        (idxMin,
                         idxMax,
                         imCMin,
                         imCMax) = man.Extrema(fMin, fMax,
                                               self.fList[i],
                                               np.array(
                                                   (np.imag(
                                                    self.epsilonList[i]),)))
                        plot1 = ax1.plot(np.real(self.fList[i]),
                                         np.real(self.epsilonList[i]), ls='-')
                        color = plot1[0].get_color()
                        ax1.plot(np.real(self.fList[i]),
                                 np.imag(self.epsilonList[i]),
                                 ls='--', color=color)
                        ax1.set_ylim(min(reCMin, imCMin), max(reCMax, imCMax))
                        (idxMin,
                         idxMax,
                         reCMin,
                         reCMax) = man.Extrema(fMin, fMax,
                                               self.fList[i],
                                               np.array(
                                                   (np.abs(
                                                       self.transList[i]),)))
                        ax2.plot(np.real(self.fList[i]),
                                 np.abs(self.transList[i]), ls='-')
                        ax2.set_ylim(reCMin, reCMax)
                        # (idxMin,
                        #  idxMax,
                        #  reCMin,
                        #  reCMax) = man.Extrema(fMin, fMax,
                        #                        self.fList[i],
                        #                        np.array(
                        #                            ((
                        #                                self.lossList[i]),)))
                        # ax3.plot(self.f, (self.lossList[i]), ls='-')
                        # ax3.set_ylim(reCMin, reCMax)
                        if fmt == 'TW':
                                specToPlotOFF = np.abs(self.ERefList[i] *
                                                       1j * self.fList[i])
                                specToPlotON = np.abs(self.Elist[i] *
                                                      1j * self.fList[i])
                        elif fmt != 'TW':
                                specToPlotOFF = np.abs(self.ERefList[i])
                                specToPlotON = np.abs(self.EList[i])
                        ax3.semilogy(np.real(self.fList[i]),
                                     specToPlotOFF * 1e3,
                                     color=color, ls='--', label=str(i))
                        ax3.semilogy(np.real(self.fList[i]),
                                     specToPlotON * 1e3,
                                     color=color, ls='-')
                        ax3.set_ylim(1e-4, 1)
                ax0.set_xlim(fMin, fMax)
                ax0.set_ylabel('$\sigma$')
                ax1.yaxis.set_label_position("right")
                ax1.yaxis.tick_right()
                ax1.set_ylabel('$\\varepsilon$')
                ax2.set_ylabel('T')
                ax2.set_xlabel('$\\nu$(THz)')
                # ax3.set_ylabel('Im{$\\frac{-1}{\\varepsilon}$}')
                ax3.set_ylabel('$E_{\mathrm{THz}}(a.u.)$')
                ax3.yaxis.set_label_position("right")
                ax3.yaxis.tick_right()
                ax3.set_xlabel('$\\nu$(THz)')
                plt.tight_layout()
                ax3.legend(loc='best')

                figt, ax0 = plt.subplots(nrows=1, ncols=1)
                for e in self.EtList:
                        i = np.where(e == self.EtList)[0][0]
                        plot0 = ax0.plot(self.tList[i],
                                         e / max(self.EtRefList[i]), ls='-',
                                         label=str(i))
                        color = plot0[0].get_color()
                        ax0.plot(self.tRefList[i],
                                 self.EtRefList[i] / max(self.EtRefList[i]),
                                 color=color, ls='--')
                        ax0.plot(self.tRefList[i],
                                 (self.EtRefList[i] -
                                  e) / max(self.EtRefList[i]),
                                 color=color, ls='-.')
                if self.window != '':
                        ax0.plot(self.t,
                                 mt.Function('Gauss', self.para,
                                             self.t))
                ax0.set_ylabel('E$_{\mathrm{THz}}$(a.u.)')
                ax0.set_xlabel('t(ps)')
                ax0.legend(loc='best')
                finalfig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2,
                                                    sharex='all',
                                                    squeeze=True)

                (idxMin,
                 idxMax,
                 reCMin,
                 reCMax) = man.Extrema(fMin, fMax,
                                       self.f,
                                       np.array(
                                           (np.real(
                                               self.sigma),)))
                (idxMin,
                 idxMax,
                 imCMin,
                 imCMax) = man.Extrema(fMin, fMax,
                                       self.f,
                                       np.array((np.imag(self.sigma),)))
                ax0.errorbar(np.real(self.f), np.real(self.sigma),
                             self.sigmaUncReal, marker='o', ls='')
                ax0.set_xlim(fMin, fMax)
                ax0.set_ylim(reCMin, reCMax)
                ax0.set_xlabel('$\\nu$(THz)')
                ax0.set_ylabel('$\mathcal{Re}\{\sigma\}$(S/m)')

                ax1.errorbar(np.real(self.f), np.imag(self.sigma),
                             self.sigmaUncImag, marker='o', ls='')
                ax1.set_ylim(imCMin, imCMax)
                ax1.set_xlabel('$\\nu$(THz)')
                ax1.set_ylabel('$\mathcal{Im}\{\sigma\}$(S/m)')
                ax1.yaxis.set_label_position("right")
                ax1.yaxis.tick_right()
                plt.tight_layout()

                return fig, figt, finalfig

        @staticmethod
        def Fit(x, y, err=0, model='', init=0,
                para=0, c=False, plot=False, guess=False,
                fitQty='Conductivity'):
                def ResWrap(f, paras, c=False):
                        def Residual(par, x, data=None):
                                model = mod.SwitchTemp(f, par, x, paras)
                                if not c:
                                        model = np.real(model)
                                if data is None:
                                        return model
                                dataShape = np.shape(data)

                                resid = model - data
                                if dataShape[0] <= 3:
                                        resid = model - data[0]
                                        err = data[1]
                                        resid = np.sqrt(resid**2 / err**2)
                                return resid.view(np.float)
                        return Residual
                yLabel_Map = {'Conductivity': '$\sigma$',
                              'Transmission': 'T'}
                yLabel = yLabel_Map[fitQty]
                if c:
                        err = err[0] + 1j * err[1]

                elif not c:
                        x, y = np.real(x), np.real(y)
                data = y
                par = fit.Parameters()
                if init:
                        if model == 'Drude':
                                par.add('tau', init['tau'])
                                par.add('N', init['N'] * 1e6)
                                paras = [para['mr']]
                        elif model == 'Cyclotron':
                                par.add('tau', init['tau'])
                                par.add('N', init['sigma0'])
                                par.add('fC', init['fC'])
                                paras = [0, para['B']]
                        elif model == 'CyclotronTransmission':
                                par.add('A', init['A'])
                                par.add('gamma', init['gamma'])
                                par.add('fC', init['fC'])
                                paras = []
                        elif model == 'DrudeNonP':
                                par.add('tau', init['tau'])
                                par.add('N', init['N'] * 1e6)
                                paras = [para['mr'], para['Eg']]
                        elif model == 'Line':
                                par.add('A', init['A'])
                                par.add('B', init['B'])
                                paras = [0]
                        else:
                                model = 'Drude'
                                par.add('tau', init['tau'])
                                par.add('N', init['N'] * 1e6)
                                paras = [para['mr']]
                                wn.warn('Warning:: Model undefined or' +
                                        ' not understood, Drude model ' +
                                        'chose as degfault', RuntimeWarning)
                guessed = mod.SwitchTemp(model, par, x, paras)
                if np.any(err):
                        data = np.append(data, err, axis=0)
                        data = np.reshape(data, (2, len(y)))
                res = ResWrap(model, paras, c)
                out = fit.minimize(res, par, args=(x,), kws={'data': data},
                                   nan_policy='omit')
                fitted = mod.SwitchTemp(model, out.params, x, paras)

                if plot:
                        print(fit.fit_report(out))
                        if c:
                                col = 2
                        elif not c:
                                col = 1
                        fitFig, axes = plt.subplots(nrows=1,
                                                    ncols=col,
                                                    sharex='all',
                                                    squeeze=True)

                        if c:
                                ebar = axes[0].errorbar(x,
                                                        np.real(y),
                                                        np.real(err),
                                                        marker='o', ls='')
                                c0 = ebar[0].get_color()
                                axes[0].plot(x, np.real(fitted),
                                             ls='-', marker='',
                                             color=c0)
                                if guess:
                                        axes[0].plot(x, np.real(guessed),
                                                     ls='--',
                                                     marker='', color=c0)
                                axes[1].errorbar(x, np.imag(y), np.imag(err),
                                                 marker='o', ls='', color=c0)
                                axes[1].plot(x, np.imag(fitted),
                                             ls='-', marker='', color=c0)
                                if guess:
                                        axes[1].plot(x, np.imag(guessed),
                                                     ls='--', marker='',
                                                     color=c0)
                                axes[0].set_xlabel('$\\nu$(THz)', x=1.05)
                                axes[0].set_ylabel(yLabel)
                                axes[1].yaxis.set_label_position('right')
                                axes[1].tick_params(which='major', right=True,
                                                    left=False)
                                axes[1].yaxis.set_ticks_position('right')

                        elif not c:
                                ebar = axes.errorbar(x,
                                                     np.real(y),
                                                     np.real(err),
                                                     marker='o', ls='')
                                c0 = ebar[0].get_color()
                                axes.plot(x, np.real(fitted),
                                          ls='-', marker='',
                                          color=c0)
                                if guess:
                                        axes.plot(x, np.real(guessed),
                                                  ls='--',
                                                  marker='', color=c0)
                                axes.set_xlabel('$\\nu$(THz)')
                                axes.set_ylabel(yLabel)
                return out.params
