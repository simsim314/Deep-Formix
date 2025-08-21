// src/compact_pos_record.cu
#include "chess.h"
#include "utils.h"
bool CompactPosRecord::encodePos(HexaBitBoardPosition *pos, uint64 val, uint32 nextIndex)
{
	chance      = pos->chance;
	whiteCastle = pos->whiteCastle;
	blackCastle = pos->blackCastle;
	enPassent   = pos->enPassent;
	perftVal    = val;
	nextLow     = nextIndex & 0x7FF;
	nextHigh    = (nextIndex >> 11) & 0x7FFFF;

	// huffman encode the position
	BoardPosition pos88;
	Utils::boardHexBBTo088(&pos88, pos);

	int i, j;
	int index088 = 0;
	int bitIndex = 0;

	for (i = 7; i >= 0; i--)
	{
		for (j = 0; j < 8; j++)
		{
			uint8 code = pos88.board[index088];
			uint8 c     = COLOR(code);
			uint8 piece = PIECE(code);
			switch (piece)
			{
			case EMPTY_SQUARE:
				writeBit(bitIndex++, 0);
				break;
			case PAWN:
				writeBit(bitIndex++, 1);
				writeBit(bitIndex++, 0);
				writeBit(bitIndex++, c);
				break;
			case KNIGHT:
				writeBit(bitIndex++, 1);
				writeBit(bitIndex++, 1);
				writeBit(bitIndex++, 0);
				writeBit(bitIndex++, 1);
				writeBit(bitIndex++, c);
				break;
			case BISHOP:
				writeBit(bitIndex++, 1);
				writeBit(bitIndex++, 1);
				writeBit(bitIndex++, 0);
				writeBit(bitIndex++, 0);
				writeBit(bitIndex++, c);
				break;
			case ROOK:
				writeBit(bitIndex++, 1);
				writeBit(bitIndex++, 1);
				writeBit(bitIndex++, 1);
				writeBit(bitIndex++, 0);
				writeBit(bitIndex++, c);
				break;
			case QUEEN:
				writeBit(bitIndex++, 1);
				writeBit(bitIndex++, 1);
				writeBit(bitIndex++, 1);
				writeBit(bitIndex++, 1);
				writeBit(bitIndex++, 0);
				writeBit(bitIndex++, c);
				break;
			case KING:
				writeBit(bitIndex++, 1);
				writeBit(bitIndex++, 1);
				writeBit(bitIndex++, 1);
				writeBit(bitIndex++, 1);
				writeBit(bitIndex++, 1);
				writeBit(bitIndex++, c);
				break;
			}
			index088++;
		}
		// skip 8 cells of padding
		index088 += 8;
	}

	// overflow!
	if (bitIndex > 164)
	{
		memset(this, 0, sizeof(this));
		return false;
	}

	return true;
}

bool CompactPosRecord::decodePos(HexaBitBoardPosition *pos, uint64 *val, uint32 *nextIndex)
{
	// huffman decode position:
	int i, j;
	int index088 = 0;
	int bitIndex = 0;
	BoardPosition pos88;

	for (i = 7; i >= 0; i--)
	{
		for (j = 0; j < 8; j++)
		{
			uint8 code;
			uint8 color;
			uint8 piece;

			int bit = getBit(bitIndex++);
			if (bit == 0)
			{   // 0
				code = EMPTY_SQUARE;
			}
			else
			{   // 1
				bit = getBit(bitIndex++);
				if (bit == 0)
				{   // 10c -> PAWN
					piece = PAWN;
					color = getBit(bitIndex++);
					code = COLOR_PIECE(color, piece);
				}
				else
				{
					// 11
					bit = getBit(bitIndex++);
					if (bit == 0)
					{
						// 110 -> knight/bishop
						bit = getBit(bitIndex++);
						if (bit == 0)
						{
							// 1100c -> bishop
							piece = BISHOP;
							color = getBit(bitIndex++);
							code = COLOR_PIECE(color, piece);
						}
						else
						{
							// 1101c -> knight
							piece = KNIGHT;
							color = getBit(bitIndex++);
							code = COLOR_PIECE(color, piece);
						}
					}
					else
					{   // 111
						bit = getBit(bitIndex++);
						if (bit == 0)
						{
							// 1110c -> rook
							piece = ROOK;
							color = getBit(bitIndex++);
							code = COLOR_PIECE(color, piece);
						}
						else
						{
							// 1111
							bit = getBit(bitIndex++);
							if (bit == 0)
							{
								// 11110c -> Queen
								piece = QUEEN;
								color = getBit(bitIndex++);
								code = COLOR_PIECE(color, piece);
							}
							else
							{
								// 11111c -> King
								piece = KING;
								color = getBit(bitIndex++);
								code = COLOR_PIECE(color, piece);
							}
						}
					}
				}
			}

			pos88.board[index088] = code;

			index088++;
		}
		// skip 8 cells of padding
		index088 += 8;
	}

	Utils::board088ToHexBB(pos, &pos88);

	pos->chance = chance;
	pos->whiteCastle = whiteCastle;
	pos->blackCastle = blackCastle;
	pos->enPassent = enPassent;
	*val = perftVal;
	*nextIndex = (nextHigh << 11) | nextLow;

	return true;
}
