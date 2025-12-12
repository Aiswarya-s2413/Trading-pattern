from rest_framework import serializers
from marketdata.models import Symbol, Index


class SymbolListItemSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    symbol = serializers.CharField()
    name = serializers.CharField()
    type = serializers.CharField()
    sector = serializers.CharField(allow_null=True)
    sector_id = serializers.IntegerField(allow_null=True)
