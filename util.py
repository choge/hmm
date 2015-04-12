#!/usr/bin/python3
# -*- encodeing: utf-8 -*-

# util library
import xml.etree.ElementTree as et
import numpy as np

def load_ghmmxml(filename):
    """Load an xml file of GHMM, in which each state,
    transition and emission probabilities are written"""
    tree = et.parse(filename)
    root = tree.getroot()
    symbols = [symbol.get('code') for symbol in root.findall('HMM/alphabet/symbol')]
    transitions = []
    emissions = []
    initials = []
    # Emission and Initial probabilities
    for state in root.findall('HMM/state'):
        initials.append(float(state.get('initial')))
        emission_prob = float_array(state.find('discrete').text)
        if len(emission_prob) != len(symbols):
            raise ValueError('failed convert emission probabilities')
        emissions.append(emission_prob)
    # Transition probabilities
    for state_id in range(len(emissions)):
        transition_dic = {}
        for transition in root.findall("HMM/transition[@source='" + str(state_id) +"']"):
            target = int(transition.get('target'))
            prob = float(transition.find('probability').text)
            transition_dic[target] = prob
        transition_prob = np.array(
                [transition_dic.get(i, 0.0) for i in range(len(emissions))],
                dtype=float)
        transitions.append(transition_prob)

    return (np.array(transitions, dtype=float),
            np.array(emissions, dtype=float),
            np.array(initials, dtype=float))


def float_array(str_exp, sep=','):
    """given a string, returns an array of float"""
    elems = str_exp.split(sep)
    try:
        return np.array(
                [float(elem) for elem in elems],
                dtype=float)
    except ValueError as e:
        e.message += "\nstr_expression = {0}".format(str_exp)

